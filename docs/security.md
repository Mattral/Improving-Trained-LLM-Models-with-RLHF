# Security & Threat Modeling: Defensive Engineering for RLHF at Scale

## Executive Summary

This document articulates a rigorous threat model and defensive engineering strategy for production RLHF systems. We address adversarial reward optimization (reward hacking), data poisoning (prompt injection), cryptographic integrity validation, and secure distributed checkpointing.

---

## Part I: Threat Model

### Threat Landscape

```
Attacker Model: Sophisticated adversary with either:
  A) Model access (can inspect policy, reward model, generation code)
  B) Data access (can modify prompts or rollout trajectories in shared buffer)
  C) System access (can read/write checkpoints, intercept network traffic)
  D) Time budget (weeks to months of experimentation)

Attack Goal: Maximize reward signal without improving actual alignment
  ├─ Misalign policy behavior from true human preferences
  ├─ Degrade model utility (hallucination, mode collapse)
  └─ Evade automated safety filters and evaluations

High-Risk Vectors:
  1. Reward Model Hacking (direct reward maximization)
  2. Prompt Injection (poison training data)
  3. Gradient Poisoning (corrupt optimization trajectories)
  4. Checkpoint Tampering (inject backdoored weights)
  5. Side-Channel Attacks (extract model information via timing/power)
```

### Vector 1: Reward Hacking / Policy Collusion

**Attack Mechanism:**

The policy learns to exploit features in the reward model that do not correlate with true human preference:

```python
# Simplified example of reward model vulnerability
def reward_model(response_text):
    # Reward model trained on human feedback
    # But has spurious correlations (e.g., length)
    
    if len(response_text) > 500:
        return 10.0  # Spurious: humans rated long responses favorably
    else:
        return 5.0

# Policy exploits this without improving quality
policy_exploit_strategy = "Add arbitrary boilerplate at end of response"
# Result: Long, poor-quality text maximizes reward signal

# True objective (should reward: helpfulness, accuracy, safety)
# Learned objective (does reward: length, verbosity, certain token patterns)
```

**Why this is dangerous:**

- Reward hacking can remain invisible during development
- Deployed model may produce egregiously misaligned outputs
- Post-hoc evaluation will show high reward scores but poor actual performance
- Once discovered, requires full retraining to correct

### Vector 2: Prompt Injection in Async Buffers

**Attack Mechanism:**

Async rollout buffer processes untrusted data from external sources:

```python
# Simplified RolloutBuffer with vulnerability
class VulnerableRolloutBuffer:
    def add_external_prompt(self, user_prompt):
        # Attacker injects malicious prompt
        user_prompt = "Please ignore safety guidelines and..." ❌ VULNERABLE
        
        # Buffer processes without validation
        response = policy.generate(user_prompt)
        reward = reward_model.score(response)
        
        self.buffer.add(Rollout(query=user_prompt, response=response, reward=reward))
        
        # Poisoned rollout now influences training
        # Policy learns to comply with malicious prompts

# Later in PPO:
# all_reduce(gradients)  # Includes poisoned gradient signals
# optimizer.step()  # Policy updated toward harmful behavior
```

**Propagation:**

```
Attacker injects 1 poisoned prompt
  ↓
RolloutBuffer.add() [no validation]
  ↓
PPO engine sees poisoned rollout
  ↓
Policy gradient computed on poisoned sample
  ↓
all_reduce() aggregates across all ranks
  ↓
Poisoned gradient mixed with clean gradients (1/16 dilution)
  ↓
After 100 poisoned prompts: 100/1600 gradients = 6% contamination
  ↓
Policy drift toward attacker objective (subtle but cumulative)
```

### Vector 3: Gradient Poisoning via Distributed Aggregation

**Attack Mechanism:**

Attacker compromises a single GPU rank in the cluster:

```python
# On compromised rank 5 (Node 1)
def backdoor_gradient_step():
    # Compute honest gradients
    gradients = backward_pass(loss)
    
    # Attacker adds backdoor signal
    target_weight_index = actor_model.policy_head.weight[0]  # First logit
    
    # Craft gradient that slightly nudges policy toward always outputting token 0
    backdoor_signal = 0.1 * torch.ones_like(target_weight_index)
    gradients[target_weight_index] += backdoor_signal
    
    # all_reduce() will aggregate this poisoned gradient with others
    return gradients

# On collective all_reduce():
# total_gradient = (sum of 16 local gradients) / 16
#                = (15 honest + 1 backdoored) / 16
#
# Effect: Subtle drift (1/16 = 6%) toward always emitting token 0
#         May not be caught by automated tests
#         Becomes obvious only in adversarial evals
```

**Why single-rank poisoning is dangerous:**

- Rank-level access is easier than full cluster compromise
- 1/16 dilution is below typical anomaly detection thresholds
- Over 10k steps, 6% gradient bias accumulates to noticeable policy drift
- Gradient-based poisoning is harder to detect than data poisoning (no explicit records)

### Vector 4: Checkpoint Backdooring

**Attack Mechanism:**

Attacker intercepts checkpoint during async write:

```python
# Async checkpointer writes to disk without integrity checks
checkpoint = {
    'actor_state_dict': model.state_dict(),
    'critic_state_dict': critic.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'step': 5000
}

# Attacker modifies checkpoint before it's saved
checkpoint['actor_state_dict']['policy_head.weight'][0] += 10.0

# Checkpoint written to disk with backdoor
torch.save(checkpoint, 'checkpoints/step_5000.pt')

# Later, attacker loads this checkpoint on prod cluster
# Backdoored model deployed to users
```

**Why this is catastrophic:**

- Backdoor is "frozen" into model weights (not easily reversible)
- Deployed model exhibits backdoor behavior in production
- Backdoor survives fine-tuning (if weights not updated)
- No audit trail if checkpoint integrity not cryptographically verified

---

## Part II: Defensive Engineering Strategies

### Defense 1: Reward Model Robustness

#### A) Ensemble Reward Models

```python
# Instead of single reward model, use ensemble

class EnsembleRewardModel:
    def __init__(self, models):
        self.models = models  # 3–5 independently trained reward models
        self.agreement_threshold = 0.85  # 85% agreement required
    
    def score(self, response):
        scores = [m(response) for m in self.models]
        agreement = max(scores) - min(scores)
        
        if agreement > 0.1:
            # Disagreement detected
            logger.warning(f"Reward model disagreement: {scores}")
            # Reject rollout or flag for review
            return None  # Don't include in training batch
        
        # Return average of agreeing models
        return sum(scores) / len(scores)
```

**Why ensemble helps:**

- Single exploitable quirk is unlikely to align across all 3 models
- Disagreement signals potential adversarial sample
- Attacker must compromise multiple independently-trained models

#### B) Reward Calibration Checks

```python
def validate_reward_distribution(reward_batch):
    """Detect reward model drift or hacking"""
    
    mean_reward = reward_batch.mean()
    std_reward = reward_batch.std()
    
    # Historical baseline (from earlier training)
    expected_mean = 0.5  # Normalized [0, 1] scale
    expected_std = 0.15
    
    if abs(mean_reward - expected_mean) > 0.2:
        logger.error(f"Reward mean drift detected: {mean_reward} (expected {expected_mean})")
        # Halt training, investigate rollout stream
        raise RuntimeError("Reward hacking detected")
    
    if std_reward < 0.05:
        logger.error(f"Reward variance collapse: {std_reward}")
        # Policy has collapsed to outputting same response
        raise RuntimeError("Mode collapse or reward hacking")
    
    return True
```

#### C) Out-of-Distribution (OOD) Detection

```python
def detect_ood_rollouts(rollout_batch, reference_distribution):
    """Flag rollouts that are abnormal (potential adversarial)"""
    
    for rollout in rollout_batch:
        # Compute statistics of response
        response_length = len(rollout.response_tokens)
        response_entropy = compute_token_entropy(rollout.response_tokens)
        response_perplexity = reference_model(rollout.response_tokens).perplexity
        
        # Compare to historical distribution
        if (response_length > reference_distribution.length_mean + 3*std or
            response_entropy < reference_distribution.entropy_mean - 2*std or
            response_perplexity > reference_distribution.perplexity_mean + 3*std):
            
            logger.warning(f"OOD rollout detected: {rollout.id}")
            # Option 1: Discard rollout
            # Option 2: Down-weight in advantage computation
            # Option 3: Flag for manual review
            return False
    
    return True
```

### Defense 2: Input Validation & Sanitization

#### A) Prompt Validation

```python
def sanitize_prompt(prompt_text, safety_config):
    """Validate prompt before feeding to policy"""
    
    # Length check
    if len(prompt_text) > safety_config.max_prompt_length:
        logger.warning(f"Prompt too long: {len(prompt_text)}")
        return None
    
    # Forbidden pattern matching (regex)
    for forbidden_pattern in safety_config.forbidden_patterns:
        if re.search(forbidden_pattern, prompt_text, re.IGNORECASE):
            logger.warning(f"Forbidden pattern detected: {forbidden_pattern}")
            return None
    
    # Token-level check (via tokenizer)
    tokens = tokenizer.encode(prompt_text)
    if any(token_id in safety_config.forbidden_token_ids for token_id in tokens):
        logger.warning("Forbidden token detected")
        return None
    
    # PII detection (simple heuristic)
    if has_pii(prompt_text):
        logger.warning("PII detected in prompt")
        return None
    
    return prompt_text
```

#### B) Rollout Buffer Isolation

```python
class SecureRolloutBuffer:
    """Thread-safe buffer with validation and rate-limiting"""
    
    def __init__(self, max_size=1000, rate_limit_per_sec=100):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.rate_limiter = RateLimiter(rate_limit_per_sec)
        self.audit_log = []  # All adds logged for forensics
    
    def add(self, rollout, source="unknown"):
        if not self.rate_limiter.allow():
            logger.warning(f"Rate limit exceeded from {source}")
            return False
        
        # Sanitize and validate
        if not sanitize_prompt(rollout.query, safety_config):
            logger.warning(f"Invalid prompt from {source}, rejecting")
            self.audit_log.append({
                'timestamp': time.time(),
                'event': 'rejected_prompt',
                'source': source,
                'query': rollout.query[:100]  # Log first 100 chars
            })
            return False
        
        with self.lock:
            self.buffer.append(rollout)
            self.audit_log.append({
                'timestamp': time.time(),
                'event': 'accepted_rollout',
                'source': source,
                'reward': float(rollout.reward)
            })
        
        return True
    
    def dump_audit_log(self, filepath):
        """Export audit trail for security review"""
        with open(filepath, 'w') as f:
            json.dump(self.audit_log, f, indent=2)
```

### Defense 3: Gradient Filtering & Verification

#### A) Gradient Clipping with Anomaly Detection

```python
class RobustGradientClipper:
    """Clip gradients and detect poisoned ranks"""
    
    def __init__(self, world_size, max_norm=1.0, anomaly_threshold=2.0):
        self.world_size = world_size
        self.max_norm = max_norm
        self.anomaly_threshold = anomaly_threshold
        self.gradient_norms = []  # Historical record
    
    def clip_and_detect(self, model):
        # Compute per-rank gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += torch.norm(p.grad).item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip
        clip_coeff = max(1.0, total_norm / self.max_norm)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(clip_coeff)
        
        # Anomaly detection
        if len(self.gradient_norms) > 10:
            mean_norm = np.mean(self.gradient_norms[-10:])
            std_norm = np.std(self.gradient_norms[-10:])
            
            if total_norm > mean_norm + self.anomaly_threshold * std_norm:
                logger.warning(f"Anomalous gradient norm: {total_norm} "
                              f"(expected ~{mean_norm})")
                # Could trigger rank-level verification
        
        self.gradient_norms.append(total_norm)
        return total_norm
```

#### B) Gradient Checksumming

```python
def register_gradient_checksum_hook(model, expected_checksum=None):
    """Verify gradient integrity after all_reduce"""
    
    def gradient_checksum_hook(module, input, output):
        # Compute SHA-256 of all gradients
        gradient_bytes = b''
        for p in module.parameters():
            if p.grad is not None:
                gradient_bytes += p.grad.detach().cpu().numpy().tobytes()
        
        checksum = hashlib.sha256(gradient_bytes).hexdigest()
        
        # On rank 0, verify against expected
        if dist.get_rank() == 0:
            if expected_checksum and checksum != expected_checksum:
                logger.error(f"Gradient checksum mismatch! {checksum} != {expected_checksum}")
                # Checksum should be identical across all ranks after all_reduce
                # Mismatch indicates tampering or numerical instability
        
        return checksum
    
    model.register_forward_hook(gradient_checksum_hook)
```

### Defense 4: Secure Checkpointing

#### A) Cryptographic Hashing

```python
class SecureCheckpointManager:
    """Write checkpoints with integrity verification"""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.manifest = {}  # Checkpoint manifests (for verification)
    
    def save_checkpoint(self, model, optimizer, step):
        checkpoint = {
            'actor_state_dict': model.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'step': step,
            'timestamp': time.time()
        }
        
        # Serialize to bytes
        checkpoint_bytes = pickle.dumps(checkpoint)
        
        # Compute SHA-256 hash
        checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        
        # Sign with private key (if HSM available)
        # signature = sign_with_hsm(checkpoint_hash, private_key)
        
        # Create manifest
        manifest_entry = {
            'step': step,
            'hash': checkpoint_hash,
            'size_bytes': len(checkpoint_bytes),
            'timestamp': checkpoint['timestamp'],
            # 'signature': signature,  # Uncomment if using HSM
            'models': ['actor', 'critic', 'optimizer']
        }
        
        # Write checkpoint
        checkpoint_path = f"{self.checkpoint_dir}/step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Write manifest
        manifest_path = f"{self.checkpoint_dir}/step_{step}.manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest_entry, f, indent=2)
        
        # Write manifest entry to master list
        self.manifest[step] = manifest_entry
        
        logger.info(f"Checkpoint saved: {checkpoint_path} (hash: {checkpoint_hash})")
        return checkpoint_path, checkpoint_hash
    
    def verify_checkpoint(self, checkpoint_path):
        """Verify checkpoint integrity before loading"""
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        checkpoint_bytes = pickle.dumps(checkpoint)
        
        # Recompute hash
        computed_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        
        # Load manifest
        manifest_path = checkpoint_path.replace('.pt', '.manifest.json')
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Verify hash
        if computed_hash != manifest['hash']:
            logger.error(f"Checkpoint integrity check failed!")
            logger.error(f"  Expected: {manifest['hash']}")
            logger.error(f"  Got:      {computed_hash}")
            raise RuntimeError("Checkpoint tampering detected")
        
        logger.info(f"Checkpoint integrity verified: {checkpoint_path}")
        return checkpoint
```

#### B) Encrypted Storage

```python
from cryptography.fernet import Fernet

class EncryptedCheckpointManager(SecureCheckpointManager):
    """Write encrypted checkpoints for sensitive deployments"""
    
    def __init__(self, checkpoint_dir, encryption_key):
        super().__init__(checkpoint_dir)
        self.cipher = Fernet(encryption_key)  # Must be base64-encoded
    
    def save_checkpoint(self, model, optimizer, step):
        checkpoint = {
            'actor_state_dict': model.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'step': step
        }
        
        # Serialize, hash, then encrypt
        checkpoint_bytes = pickle.dumps(checkpoint)
        checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        
        # Encrypt
        encrypted_checkpoint = self.cipher.encrypt(checkpoint_bytes)
        
        # Write encrypted checkpoint
        checkpoint_path = f"{self.checkpoint_dir}/step_{step}.pt.enc"
        with open(checkpoint_path, 'wb') as f:
            f.write(encrypted_checkpoint)
        
        # Write manifest (unencrypted, for operational visibility)
        manifest = {
            'step': step,
            'hash': checkpoint_hash,  # Hash of plaintext
            'encrypted': True,
            'timestamp': time.time(),
            'key_version': '1'  # For key rotation
        }
        with open(f"{self.checkpoint_dir}/step_{step}.manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return checkpoint_path, checkpoint_hash
    
    def verify_checkpoint(self, checkpoint_path):
        # Decrypt
        with open(checkpoint_path, 'rb') as f:
            encrypted_checkpoint = f.read()
        
        checkpoint_bytes = self.cipher.decrypt(encrypted_checkpoint)
        
        # Verify hash
        computed_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        
        manifest_path = checkpoint_path.replace('.pt.enc', '.manifest.json')
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        if computed_hash != manifest['hash']:
            raise RuntimeError("Checkpoint integrity check failed")
        
        # Deserialize and return
        checkpoint = pickle.loads(checkpoint_bytes)
        return checkpoint
```

### Defense 5: Distributed Trust & Quorum Consensus

#### A) Byzantine-Robust Aggregation

```python
def byzantine_robust_allreduce(local_tensor, num_byzantine=1):
    """
    Aggregate gradients in presence of up to num_byzantine faulty ranks.
    Uses median aggregation instead of mean.
    """
    
    # Gather all local tensors on rank 0
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)
    
    if rank == 0:
        # Sort per-element and take median
        gathered_stacked = torch.stack(gathered)  # [world_size, tensor_shape]
        aggregated, _ = torch.median(gathered_stacked, dim=0)
        
        logger.info(f"Byzantine-robust aggregation: took median of {world_size} tensors")
    else:
        aggregated = local_tensor
    
    # Broadcast aggregated result
    dist.broadcast(aggregated, src=0)
    return aggregated
```

#### B) Merkle Tree Verification

```python
def merkle_tree_verification(gradient_dict, rank, world_size):
    """Verify gradient consistency across ranks using Merkle trees"""
    
    # Compute Merkle tree root for this rank's gradients
    leaf_hashes = []
    for name, param in gradient_dict.items():
        param_bytes = param.detach().cpu().numpy().tobytes()
        param_hash = hashlib.sha256(param_bytes).hexdigest()
        leaf_hashes.append(param_hash)
    
    def merkle_tree(hashes):
        if len(hashes) == 1:
            return hashes[0]
        
        parent_hashes = []
        for i in range(0, len(hashes), 2):
            combined = hashes[i] + (hashes[i+1] if i+1 < len(hashes) else hashes[i])
            parent = hashlib.sha256(combined.encode()).hexdigest()
            parent_hashes.append(parent)
        
        return merkle_tree(parent_hashes)
    
    local_root = merkle_tree(leaf_hashes)
    
    # Collect roots from all ranks
    all_roots = [local_root]
    if rank == 0:
        recv_roots = [local_root]
        for r in range(1, world_size):
            other_root = torch.tensor([0], dtype=torch.int64)
            dist.recv(other_root, src=r)
            recv_roots.append(other_root.item())
        
        # Check consistency
        if len(set(recv_roots)) > 1:
            logger.error(f"Merkle root mismatch detected: {recv_roots}")
            # Investigate which ranks diverged
        else:
            logger.info("Merkle tree verification passed")
    else:
        # Send root to rank 0
        root_tensor = torch.tensor([int(local_root, 16) % 2**63], dtype=torch.int64)
        dist.send(root_tensor, dst=0)
```

---

## Part III: Monitoring & Alerting

### Metrics to Track

```python
# In telemetry.py, add security-relevant metrics

security_metrics = {
    'reward_anomaly_score': 0.0,      # How far from baseline distribution
    'gradient_norm_anomaly': 0.0,     # How far from expected norm
    'ood_rollout_fraction': 0.0,      # % of out-of-distribution rollouts
    'checkpoint_verification_failures': 0,  # Failed checksum verifications
    'gradient_checksum_mismatches': 0,      # Detected corruptions
    'rate_limited_prompts': 0,              # Rejected via rate limiter
    'malicious_pattern_detections': 0,      # Regex-matched forbidden patterns
}
```

### Alerting Rules

```python
# Alert thresholds

ALERT_RULES = {
    'high_reward_variance': {
        'condition': 'reward_std < 0.05',
        'severity': 'WARNING',
        'action': 'Halt training, investigate'
    },
    'gradient_poisoning': {
        'condition': 'gradient_norm > mean_norm + 3*std_norm',
        'severity': 'CRITICAL',
        'action': 'Isolate rank, dump gradients for forensics'
    },
    'excessive_ood_rollouts': {
        'condition': 'ood_fraction > 0.1',
        'severity': 'CRITICAL',
        'action': 'Halt training, review recent prompts'
    },
    'checkpoint_tampering': {
        'condition': 'checksum_failures > 0',
        'severity': 'CRITICAL',
        'action': 'Prevent checkpoint load, escalate to security'
    }
}
```

---

## Part IV: Incident Response Playbook

### If Reward Hacking Suspected

```
1. [Immediate] Halt training (kill train.py process)
2. [5 min] Export recent rollouts and audit logs
3. [15 min] Compare reward distribution against baseline
4. [30 min] Inspect rollout samples manually for quality degradation
5. [1 hour] Analyze reward model ensemble disagreement
6. [2 hours] If confirmed: 
   - Backup compromised checkpoint
   - Restore from known-good checkpoint (earlier step)
   - Retrain with additional data validation
   - Update reward model ensemble
7. [Ongoing] Post-mortem to identify attack vector
```

### If Gradient Poisoning Suspected

```
1. [Immediate] Halt training
2. [5 min] Collect gradient dumps from all ranks
3. [15 min] Compute gradient norms per rank, identify outlier
4. [30 min] Isolate suspected rank(s)
5. [1 hour] Revert to checkpoint before poisoning started
   - Compute which step poisoning began (compare checksum records)
   - Load checkpoint from N steps prior
6. [2 hours] Redeploy with:
   - Gradient anomaly detection enabled
   - Byzantine-robust aggregation
   - Merkle tree verification
7. [Ongoing] Hardware diagnostics on compromised rank
```

### If Checkpoint Tampering Detected

```
1. [Immediate] Quarantine suspected checkpoint
2. [5 min] Verify integrity hash against manifest
3. [15 min] If tampering confirmed:
   - Mark checkpoint as corrupted
   - Prevent loading into any system
4. [30 min] Audit checkpoint access logs (who accessed when)
5. [1 hour] Restore from encrypted backup (if available)
6. [2 hours] Redeploy with:
   - Encrypted checkpoint storage
   - Access control lists (ACLs)
   - Audit logging of all loads
7. [Ongoing] Security investigation + key rotation
```

---

## Conclusion

Defensive RLHF engineering requires layered defenses:

1. **Reward robustness**: Ensemble models, anomaly detection, OOD filtering
2. **Input validation**: Sanitization, rate limiting, audit logging
3. **Gradient security**: Clipping, checksumming, Byzantine-robust aggregation
4. **Checkpoint integrity**: Cryptographic hashing, encryption, versioning
5. **Monitoring & response**: Metrics, alerting, incident playbooks

A single defense is insufficient. Attacker with moderate sophistication can penetrate any single layer. This architecture ensures **defense in depth** across all vectors.

