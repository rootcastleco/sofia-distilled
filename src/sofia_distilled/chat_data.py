"""Original English prompt bank; teacher responses are generated during training."""

TOPICS = (
    "root mean square of an acceleration waveform",
    "crest factor and impulsive vibration",
    "kurtosis and its limitations for fault detection",
    "the Nyquist sampling criterion and aliasing",
    "FFT resolution and window duration",
    "spectral leakage and Hann windows",
    "Welch power spectral density estimation",
    "Hilbert envelope analysis for bearing impulses",
    "shaft-speed normalization in vibration analysis",
    "the distinction between imbalance and misalignment signatures",
    "why a synthetic fault classifier needs real-world validation",
    "sensor mounting and accelerometer calibration",
    "the difference between acceleration, velocity, and displacement",
    "train-validation-test splits and data leakage",
    "normalizing features with training-set statistics only",
    "softmax probability versus calibrated confidence",
    "temperature scaling in knowledge distillation",
    "KL divergence between teacher and student probabilities",
    "cross-entropy for multiclass classification",
    "LoRA and low-rank parameter adaptation",
    "layer pruning in a transformer language model",
    "why a small distillation run is not foundation-model pretraining",
    "SHA-256 checksums for model artifacts",
    "safetensors and the risks of loading untrusted pickle files",
    "NumPy broadcasting for vectorized feature calculations",
    "finite-value validation in sensor pipelines",
    "Python context managers for reading files",
    "the difference between a list and a NumPy array",
    "unit tests for numerical algorithms",
    "JSON serialization of floating-point measurements",
    "MQTT quality-of-service levels",
    "Modbus registers and endianness",
    "bounded buffers in edge telemetry systems",
    "monotonic clocks versus wall clocks",
    "why a language model should not directly actuate machinery",
    "default-deny authorization for control commands",
    "harmonic distortion in electrical measurements",
    "power factor and active versus apparent power",
    "three-phase voltage imbalance",
    "thermal sensor drift and trend analysis",
    "the exact classical formula for a product-rotation quantum kernel",
    "shared terminal-unitary invariance of state overlaps",
    "global depolarization and affine Hilbert-Schmidt kernels",
    "matched ridge regularization under affine kernel rescaling",
    "finite-shot SWAP measurement uncertainty",
    "the difference between useful kernel geometry and quantum computational advantage",
    "training-only centering in kernel ridge regression",
    "why numerical identity checks do not prove hardware quantum advantage",
)

STYLES = (
    "Explain {topic} in three concise sentences for a junior engineer.",
    "What is {topic}? Give a short practical example.",
    "Describe one common mistake involving {topic} and how to avoid it.",
    "Give a brief technical explanation of {topic}. State an important limitation.",
)

SYSTEM = (
    "You are Sofia, an experimental technical assistant by Rootcastle Engineering. "
    "Answer in concise English. Explain engineering and Python concepts accurately. "
    "Distinguish measured evidence from assumptions."
)

# Original concise reference answers. These are not copied from the teacher.
# Four styles share each topic reference; entire topic families stay in one split.
REFERENCES = (
    "RMS is sqrt(mean(x**2)) for sampled acceleration x. It measures signal magnitude, not the signed mean; a sine wave of amplitude A has RMS A/sqrt(2). State the acceleration unit and the time interval used.",
    "Crest factor is max(abs(x))/RMS(x). A large value can indicate sparse impulses, but noise, saturation and window duration also affect it. It does not identify a bearing fault by itself.",
    "Pearson kurtosis is mean((x-mean(x))**4)/variance(x)**2. A Gaussian distribution has kurtosis 3; excess kurtosis subtracts 3. Large values suggest heavy tails or impulses, but do not establish the physical cause.",
    "A band-limited signal requires a sample rate greater than twice its highest frequency for ideal reconstruction. An analog anti-alias filter must attenuate components above Nyquist before sampling. Aliased energy cannot generally be removed afterward using a digital filter.",
    "FFT bin spacing is sample_rate/N, or 1/window_duration. A longer measured window gives finer frequency spacing. Zero padding interpolates the spectrum without creating new physical frequency resolution.",
    "Spectral leakage occurs when a finite observation spreads a tone across frequency bins. A Hann window reduces sidelobes but broadens the main lobe and changes amplitude gain. Apply the appropriate window normalization before comparing amplitudes.",
    "Welch PSD averages periodograms of overlapping windowed segments. Averaging reduces estimator variance at the cost of segment-limited frequency resolution. Use power-per-hertz units and correct sampling-rate and window-energy normalization.",
    "Envelope analysis forms an analytic signal and examines its amplitude envelope. A selected resonance band can expose repeated modulation from impulses. Filter choice and speed variation matter; an envelope peak alone is not a confirmed bearing defect.",
    "Order tracking expresses frequency relative to shaft rotation frequency. A tone at twice shaft speed is order 2. Variable-speed data need suitable resampling or tracking; simply dividing one FFT axis by an average speed may smear features.",
    "Imbalance often emphasizes the first shaft harmonic, while misalignment can produce higher harmonics and axial vibration. These are tendencies, not unique signatures. Mounting, load, machine geometry and physical inspection are needed for diagnosis.",
    "Synthetic data test a specified generator rather than the full distribution of real machines. High simulated accuracy can reflect easy class formulas or missing confounders. Evaluate independent recorded signals, machine-level splits and acquisition conditions before deployment.",
    "Accelerometer mounting changes the measured transfer function and usable bandwidth. Calibration relates sensor output to a known physical acceleration. Record orientation, mounting, units, sample rate and clipping status before comparing measurements.",
    "Acceleration is the time derivative of velocity, and velocity is the derivative of displacement. For a sinusoid, their amplitude factors differ by angular frequency. Integration needs careful treatment of offsets and low-frequency drift.",
    "Use training data to fit model parameters, validation data to choose settings, and a held-out test set for final reporting. Split related machines or topic families together to reduce leakage. Repeated test-driven tuning invalidates the intended hold-out comparison.",
    "Fit feature means and scales using the training partition only. Apply those same statistics to validation, test and deployment data. Computing the scaler from all observations leaks information about the evaluation distribution.",
    "Softmax maps logits to nonnegative values that sum to one. A high softmax score is not evidence of calibrated confidence or correctness on unfamiliar inputs. Check calibration and distribution shift using separate evaluation data.",
    "Temperature T softens teacher probabilities by applying softmax(logits/T). Higher T exposes probability mass outside the top class. In standard logit distillation the KL term is commonly multiplied by T squared to adjust gradient scale.",
    "KL(p_teacher || p_student) is sum(p_teacher*log(p_teacher/p_student)). It is asymmetric and compares probability distributions, not hard labels alone. Mask prompt and padding positions when distilling assistant responses.",
    "For one correct class y, cross-entropy is -log(p[y]). It penalizes confident incorrect predictions and is usually averaged over samples or valid tokens. Class imbalance and label quality affect its interpretation.",
    "LoRA trains low-rank updates to selected weight matrices while freezing the base model. This reduces trainable parameters and optimizer memory. A merged adapter changes the full checkpoint and still depends on the original pretrained weights.",
    "Layer pruning removes transformer blocks to reduce parameter count and compute. Copied embeddings and surviving blocks retain some pretrained structure. Distillation can repair part of the damage, but broad quality must be measured rather than assumed.",
    "A short distillation run adapts a student using a pretrained teacher or inherited weights. Foundation pretraining from random initialization requires a substantially broader corpus and compute budget. Report the teacher, initialization, data volume and measured scope explicitly.",
    "SHA-256 computes a digest of exact artifact bytes. Comparing a downloaded weight file with a trusted expected digest detects changes. A digest supplied by the same untrusted source does not independently authenticate authorship.",
    "Safetensors stores tensor values without the general Python object deserialization used by pickle. Avoid loading untrusted pickle checkpoints because deserialization can execute code. File integrity and trusted provenance still need separate checks.",
    "NumPy broadcasting aligns compatible array dimensions for vectorized operations. For X of shape (n,d), subtracting a mean vector of shape (d,) normalizes each row. Validate shapes because accidental broadcasting can silently compute the wrong result.",
    "Reject NaN and infinity before numerical sensor calculations. Use np.isfinite(values).all() together with shape and unit checks. Replacing invalid measurements with arbitrary zeros can hide acquisition failures.",
    "A Python with block calls a context manager's cleanup when execution leaves the block. Use with open(path) as f to close a file even if reading raises an exception. It does not make the file contents trustworthy.",
    "A Python list is a general sequence of object references; a NumPy array is a typed numerical container. NumPy arrays support vectorized arithmetic and explicit shapes. Multiplying a list repeats it, while multiplying a numeric array scales its elements.",
    "Numerical tests should compare against independent formulas, analytical special cases and meaningful tolerances. Include invalid inputs and invariants such as energy or symmetry. A test that repeats the implementation's own logic may reproduce the same mistake.",
    "JSON transports structured data but does not encode physical units automatically. Serialize finite measurements with explicit unit, timestamp and quality fields. Reject nonstandard NaN or infinity values when strict interoperable JSON is required.",
    "MQTT QoS 0 is at most once, QoS 1 is at least once, and QoS 2 is exactly once within the protocol exchange. QoS 1 can deliver duplicates. Application-level persistence and idempotency remain important for end-to-end processing.",
    "A Modbus register contains 16 bits. A multiregister value needs a specified register order, byte interpretation and data type from the device map. Guessing those conventions can yield plausible but incorrect numerical values.",
    "A bounded buffer limits the number of retained records. Choose and document overflow behavior such as drop-oldest, reject-newest or backpressure. Monitor dropped records because a memory bound alone does not preserve every measurement.",
    "A wall clock represents calendar time and can jump after synchronization. A monotonic clock measures elapsed intervals without going backward. Use monotonic time for local deadlines and record wall timestamps separately for event correlation.",
    "A language model can produce incorrect or unauthorized control instructions. Keep advisory text separate from a deterministic control interface with explicit authorization and physical interlocks. Model confidence must not bypass those controls.",
    "Default-deny authorization rejects commands unless every required permission and condition is satisfied. Check operator authorization, command allowlists, freshness and interlocks before execution. Missing context must not silently become approval.",
    "Total harmonic distortion compares the RMS contribution of higher harmonics with the fundamental RMS. State which harmonics and acquisition bandwidth are included. A distorted waveform needs adequate sampling and a well-defined fundamental estimate.",
    "Active power is the average instantaneous voltage-current product. Apparent power is Vrms*Irms, and true power factor is active power divided by apparent power. With waveform distortion, true power factor is not generally just the cosine of a phase angle.",
    "Three-phase voltage imbalance can be studied using positive- and negative-sequence components. A common ratio uses negative-sequence magnitude divided by positive-sequence magnitude. Specify the measurement definition rather than treating every imbalance metric as equivalent.",
    "Thermal sensor drift changes the measurement response over time. Compare against reference measurements and operating conditions before interpreting a trend as equipment degradation. Temperature rate estimates are sensitive to sample interval and noise.",
    "For product Ry angle encoding, the squared state-overlap kernel is product_j cos(s*(x_j-z_j)/2)**2. This expression is evaluated exactly on a classical computer in linear work per feature dimension. A useful classification score does not establish quantum computational advantage.",
    "Applying the same input-independent terminal unitary to both states preserves their inner product. It can change entanglement while leaving the overlap kernel unchanged. An input-dependent layer between operations is outside that cancellation argument.",
    "Under common global depolarization of normalized states, the Hilbert-Schmidt kernel becomes a*k+b, with a=(1-p)**2 and b=(1-a)/D. The noisy diagonal is generally below one. This identity does not model arbitrary gate-local noise or Uhlmann fidelity.",
    "Training-only centering removes the constant offset of an affine kernel transform. For a positive scale a, centred ridge predictions are preserved by scaling the ridge penalty to a*lambda. The completely depolarized endpoint a=0 does not support that inverse equivalence.",
    "An ideal SWAP estimator uses binomial counts with probability (1+k)/2 and estimate 2*C/shots-1. Its variance is (1-k*k)/shots, so negative estimates can occur. Finite measurement noise does not inherit exact matched-regularization invariance.",
    "A kernel can improve prediction through its feature geometry while remaining efficiently classically evaluable. Compare it against its exact classical implementation and fair competing baselines. Quantum computational advantage needs separate resource and hardware evidence.",
    "Center the training Gram matrix using only training row and grand means. Center each new similarity row using those training statistics and that row's own mean. Estimating an eigenspace from held-out test data would leak evaluation information.",
    "Numerical identity checks compare independently implemented calculations at chosen inputs. They support implementation correctness but do not replace universal proofs or physical device validation. Report tolerances, tested dimensions, seeds and retained outputs.",
)

if len(REFERENCES) != len(TOPICS):
    raise RuntimeError("Each topic requires one original reference answer")


def prompt_bank():
    """Split whole topic families, keeping related paraphrases in one partition."""
    # Predetermined split; test topics are not used to pick hyperparameters or checkpoints.
    validation_topics = {4, 12, 25, 38}
    test_topics = {6, 19, 27, 35}
    rows = []
    for topic_id, topic in enumerate(TOPICS):
        split = (
            "validation"
            if topic_id in validation_topics
            else "test"
            if topic_id in test_topics
            else "train"
        )
        for style_id, style in enumerate(STYLES):
            rows.append(
                {
                    "id": f"topic-{topic_id:02d}-style-{style_id}",
                    "topic_id": topic_id,
                    "split": split,
                    "prompt": style.format(topic=topic),
                    "source": "original Sofia prompt bank v1",
                    "response": REFERENCES[topic_id],
                }
            )
    return rows
