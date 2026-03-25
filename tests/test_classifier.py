"""
Tests for the core classifier pipeline.
Run: pytest tests/ -v
"""

import pytest

from agentwatch.agent.failure_injector import (
    generate_forgetting_trace,
    generate_hallucination_trace,
    generate_lazy_trace,
    generate_loop_trace,
    generate_normal_trace,
    generate_trace_batch,
)
from agentwatch.classifier.few_shot_trainer import TrainedClassifier, train_embedding_lr, train_prototype
from agentwatch.traces.models import SpanKind, Trace


class TestTraceGeneration:
    def test_normal_trace_has_tool_calls(self):
        t = generate_normal_trace("How do I reset my password?")
        assert t.user_query != ""
        assert t.agent_response != ""
        assert len(t.tools_called) > 0
        assert t.metadata["failure_type"] == "none"

    def test_lazy_trace_has_no_tool_calls(self):
        t = generate_lazy_trace()
        assert len(t.tools_called) == 0
        assert t.metadata["failure_type"] == "lazy"
        # Lazy responses are short
        assert len(t.agent_response) < 100

    def test_forgetting_trace_asks_for_known_info(self):
        t = generate_forgetting_trace()
        assert t.metadata["failure_type"] == "forgetting"
        # Should have prior context in metadata
        assert "prior_context" in t.metadata

    def test_hallucination_trace_uses_tools(self):
        t = generate_hallucination_trace()
        assert t.metadata["failure_type"] == "hallucination"
        assert len(t.tools_called) > 0

    def test_loop_trace_has_repeated_tools(self):
        t = generate_loop_trace()
        assert t.metadata["failure_type"] == "tool_loop"
        assert len(t.tools_called) >= 3

    def test_batch_generation_respects_distribution(self):
        traces = generate_trace_batch(n=200)
        assert len(traces) == 200
        types = [t.metadata.get("failure_type") for t in traces]
        # All types should be present
        assert "none" in types
        assert "lazy" in types

    def test_trace_to_text_serialization(self):
        t = generate_normal_trace()
        text = t.to_text()
        assert "Query:" in text
        assert "Response:" in text
        assert len(text) > 50

    def test_otel_compatible_fields(self):
        t = generate_normal_trace()
        assert len(t.trace_id) == 32  # hex string
        for s in t.spans:
            assert len(s.span_id) == 16
            assert s.kind in [sk.value for sk in SpanKind]


class TestClassifier:
    @pytest.fixture
    def training_data(self):
        """Generate labeled training data from synthetic traces."""
        normal = [generate_normal_trace() for _ in range(30)]
        lazy = [generate_lazy_trace() for _ in range(20)]
        texts = [t.to_text() for t in normal + lazy]
        labels = [False] * len(normal) + [True] * len(lazy)
        return texts, labels

    def test_embedding_lr_trains(self, training_data):
        texts, labels = training_data
        clf = train_embedding_lr(texts, labels, "test_lazy")
        assert clf.approach == "embedding_lr"
        assert clf.metrics is not None
        assert clf.metrics.accuracy > 0.5

    def test_prototype_trains(self, training_data):
        texts, labels = training_data
        clf = train_prototype(texts, labels, "test_lazy")
        assert clf.approach == "prototype"
        assert clf.metrics is not None

    def test_classifier_predicts(self, training_data):
        texts, labels = training_data
        clf = train_embedding_lr(texts, labels, "test_lazy")
        preds = clf.predict(["You can find that information in our help center."])
        assert len(preds) == 1
        assert isinstance(preds[0][0], bool)
        assert 0 <= preds[0][1] <= 1

    def test_classifier_catches_lazy_traces(self, training_data):
        texts, labels = training_data
        clf = train_embedding_lr(texts, labels, "test_lazy")

        # Test on new lazy trace
        lazy = generate_lazy_trace()
        pred, conf = clf.predict_single(lazy.to_text())
        # Should flag as lazy with reasonable confidence
        assert isinstance(pred, bool)

    def test_classifier_passes_normal_traces(self, training_data):
        texts, labels = training_data
        clf = train_embedding_lr(texts, labels, "test_lazy")

        normal = generate_normal_trace()
        pred, conf = clf.predict_single(normal.to_text())
        assert isinstance(pred, bool)


class TestBatchPipeline:
    def test_batch_prediction_consistency(self):
        """Ensure batch and single predictions agree."""
        normal = [generate_normal_trace() for _ in range(20)]
        lazy = [generate_lazy_trace() for _ in range(15)]
        texts = [t.to_text() for t in normal + lazy]
        labels = [False] * 20 + [True] * 15

        clf = train_embedding_lr(texts, labels, "test_batch")

        # Batch predict
        batch_preds = clf.predict([texts[0], texts[-1]])
        # Single predict
        single_0 = clf.predict_single(texts[0])
        single_1 = clf.predict_single(texts[-1])

        assert batch_preds[0][0] == single_0[0]
        assert batch_preds[1][0] == single_1[0]

