"""unciv_train — REINFORCE-with-baseline training for the self-play policy net.

Trains a tiny MLP on trajectory shards emitted by the Unciv data plane, exports `policy.onnx`
(the contract the in-JVM `OnnxPolicy` consumes), and drives the round loop that produces the
win-rate-vs-RandomPolicy learning curve.
"""
