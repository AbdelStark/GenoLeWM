# What does this edit do? A genomic world model, and an honest negative result

*Building a latent world model for DNA edits, watching it lose to a one-line baseline, and finding the reason more interesting than a win would have been.*

---

A DNA language model can read a stretch of genome and tell you something useful about it: how likely the sequence is, what it resembles, where the interesting regions are. That is a *reading* task. There is a different task hiding next to it, and most foundation models do not touch it. It is a *what-if* task. Given a piece of DNA and a specific edit, change this one base, insert these three, delete that codon, what happens? Not to the letters, we already know those, but to everything the model thinks the sequence *means*.

If you could answer that cheaply, a lot of things open up at once. You could score a variant by how much it perturbs the model's internal state. You could chain edits and ask what a whole haplotype looks like without re-reading the genome each time. You could even *plan*: search over edits to reach some target state. That is the shape of a **world model**. A world model predicts the next state of the world given an action. Here the world is a window of DNA, the state is the model's internal summary of it (its *embedding*), and the action is an edit.

This post is about GenoLeWM, a small project that tried to build exactly that, and the result is negative. The trained model does not beat the baseline. I want to walk through it anyway, because the *reason* it fails is a clean, general lesson about frozen-encoder world models, and because I think a negative result you can fully reproduce is worth more than a positive one you cannot.

Everything below is bound to a single released checkpoint (`v0.2.1-r1`) and every number traces back to a content-addressed artifact. I will be specific about what is real and what is not.

## The idea, and why it is appealing

Start with the version that sounds great, because it genuinely does.

Take a pretrained DNA model. We used Carbon-500M, a 500-million-parameter decoder-only model over DNA, which reads the sequence in 6-base chunks, so a 12,288 base-pair window becomes 2,048 tokens. Feed it that reference window and pull out a single vector that summarizes the region around your edit. Call that vector `s_t`, the *state*. It lives in 1024 dimensions. You can think of it as the model's compressed opinion about that stretch of DNA.

Now **freeze** the big model. Never train it again. Instead, train a small network, around 40 million parameters, that takes two things: the state `s_t`, and an embedding of the edit you want to make (its position, its type, the reference and alternate bases). It outputs a predicted new state, `ŝ_{t+1}`. You train it by actually applying the edit, running the frozen model on the edited window to get the true new state `s_{t+1}`, and pushing the prediction toward it with a loss that penalizes both pointing the wrong way (the cosine term) and landing at the wrong magnitude (the mean-squared-error term).

```
s_t      = encode(reference window)        # frozen, expensive, done once
a        = embed(edit)                      # tiny
ŝ_{t+1}  = predictor(s_t, a)                # tiny, the only thing we train
s_{t+1}  = encode(edited window)            # the target, also from the frozen model
loss     = (1 - cos(ŝ_{t+1}, s_{t+1})) + 0.1 * MSE(ŝ_{t+1}, s_{t+1}) / d
```

This is a **Joint-Embedding Predictive Architecture**, JEPA, applied to genomic edits. The defining move of a JEPA is that you predict in the space of *embeddings*, not in the space of raw data. You never try to generate DNA letters. You try to predict where the edit lands in the encoder's latent space. That sidesteps the pain of generative modeling and keeps everything in a tidy vector space where distances are meaningful.

The economics are the selling point. The 500M-parameter model is the only heavy thing in the system. You pay for it once per reference window, cache the result, and then every downstream question is cheap latent arithmetic on the small predictor. Score a thousand variants in the same region and you pay for one expensive encode and a thousand cheap predictions. A five-edit haplotype is one encode and five small steps, with the big model never running again. Planning a set of edits to hit a target happens entirely in latent space. The phrase I kept using in my head was *pay for the encoder once, query it forever*.

There is also a subtle safety benefit to freezing, and it matters for what comes later. JEPAs have a notorious failure mode called **representation collapse**: if you train the encoder and the predictor together, the encoder can cheat by mapping everything to nearly the same vector, which makes the prediction loss trivially small and the representation useless. Whole sub-fields exist to prevent this. But if the encoder is frozen, collapse is impossible by construction. The targets are fixed. The predictor cannot drag the encoder anywhere. So freezing buys you cheap inference *and* a free pass on the stability problem that sinks most of these systems. It feels like a bargain.

Hold onto that feeling, because the bargain is the trap.

## What actually happened

We trained the predictor and ran a benchmark suite: binary variant-effect classification on ClinVar (clinically annotated human variants, labeled pathogenic or benign), correlation against a BRCA2 saturation mutagenesis assay and a regulatory-variant benchmark, and a latent rollout-fidelity test for multi-edit chains.

The honest summary is that it did not work. Here are the headline numbers, with the frozen encoder's own zero-shot score as the baseline. The encoder gets a variant score for free: the log-likelihood it assigns the reference window minus the log-likelihood it assigns the edited one, so a higher score means it finds the reference more probable than the edit. The fair question is therefore never "does this beat a clinical tool," it is "does the learned predictor add anything over the encoder it sits on." Everything below is scored by AUROC, a ranking-quality measure where 1.0 is perfect and 0.5 is a coin flip.

| Benchmark | N | GenoLeWM | Carbon zero-shot | Gap |
|---|---|---|---|---|
| ClinVar coding (AUROC) | 16 | 0.7344 | 0.9219 | −0.1875 |
| ClinVar non-coding (AUROC) | 16 | 0.5625 | 0.8750 | −0.3125 |
| BRCA2 saturation (Spearman) | 32 | 0.1492 | 0.4769 | −0.3277 |

On every metric that measures whether you can tell pathogenic from benign, the learned model is *worse* than the frozen encoder it was built on top of. Not by a rounding error. By a lot.

Two caveats before anyone over-reads this, in either direction. First, these benchmark slices are tiny. The ClinVar slices have 16 variants each, and the BRCA2 slice has 32. Every metric is quantized in steps of 1/16, so a confidence interval over them is meaningless, and the right statement is not "GenoLeWM reliably trails on every row" but "on slices this small, it shows no detectable improvement and shows clear deficits where the gap is large." Second, the point of this post is not the table. The point is the *next* result, which is small, clean, and I think actually explains the table.

## The result that explains the result

We also measured **rollout fidelity**: how well the predictor tracks reality when you chain several edits in latent space, which is the multi-step use case from the start. Apply a chain of edits, predict the final latent state, and compare it to the true latent state of the fully edited window using cosine similarity. Cosine similarity measures the angle between two vectors: 1.0 means they point the same way, 0 means they are unrelated. Across the two rollout conditions we tested, eight edit-chains each, the model scored **0.289 and 0.302**.

That number means nothing on its own, so we put it next to a baseline. Not a clever baseline. The dumbest baseline I could think of: **predict no change at all**. Just output the original state `s_t` and pretend the edit did nothing. That baseline scored **0.998 and 0.991** on the same two conditions.

A model that ignores the edit entirely, that does literally nothing, lands at cosine about 0.99 to the true post-edit state. The trained model, the one that is supposed to understand edits, lands at about 0.29, far below it.

When I first saw this I assumed a bug. It is not a bug. It is the whole story, and once you see why, the benchmark table stops being mysterious.

## The latent-residual trap

Here is the intuition, and it is worth slowing down for, because it generalizes well beyond genomics.

You have a window of about 12,000 base pairs. You change one of them. One single-nucleotide variant, one letter out of twelve thousand. Then you run a 500M-parameter model over the whole window and squeeze it down to a single 1024-dimensional vector.

How much does that vector move?

Barely at all. The window is almost identical to what it was. One base in twelve thousand is a whisper. The encoder, which was trained to represent what a sequence *is*, produces an embedding that is almost exactly the same as before. So the true new state `s_{t+1}` is sitting right next to the old state `s_t`. They are nearly the same vector.

Now look at what that does to the learning problem. The thing the predictor needs to produce is `s_{t+1}`. But `s_{t+1} ≈ s_t`. So the single best cheap answer, the one that captures almost all of the achievable similarity, is to copy the input. Output `s_t` and you are already at cosine 0.99. The actual useful signal, the part that says *what this particular edit did*, lives in the tiny difference:

```
Δ = s_{t+1} − s_t
```

`Δ` is small in magnitude, yet it is the only part that changes from one edit to the next, and it points in a direction that the bulk vector `s_t` completely dominates. It is a needle, and the haystack is the input you were handed for free.

This is the **latent-residual trap**. The quantity you actually want to predict is a small residual on top of a strong, free baseline. And the loss function knows it. Gradient descent is pulled hard toward the copy solution, because copying is most of the reward.

It gets sharper. Our predictor is built as a *residual* network with an *identity initialization*: at the start of training, before it learns anything, it outputs exactly `s_t`. In other words, **the model begins its life sitting precisely on the do-nothing baseline.** Training has to push it *off* that point, against a gradient that mostly rewards staying on it, to chase a faint residual the frozen encoder was never asked to make legible. What we ended up with is not a model that learned `Δ` and fell a little short. It is a model that learned a slightly *distorted* copy, which is why it scores *below* the clean copy. It took the one free thing and made it worse.

The rollout number and the benchmark table are the same phenomenon. You score a variant by how far the predicted state moved from the original. But if the predictor only ever emits a slightly corrupted copy of `s_t`, then "how far it moved" is measuring the corruption, not the edit. Two different variants get different scores mainly because the copy was distorted differently, not because the edits differ. That is why the rankings come out bad: they rest on a movement signal that is mostly artifact.

## The part I actually want you to remember

Go back to the bargain. We froze the encoder for two reasons: it made inference cheap, and it made collapse impossible. Both are true. But look at what freezing also did.

The reason the residual `Δ` is hard to learn is that the encoder's geometry was never shaped to make edits legible. Carbon was trained to model what sequences *are*, to assign likelihoods, to represent context. Nothing in its training ever asked it to arrange its latent space so that "the effect of an edit" is a clean, low-dimensional, roughly linear direction. There is no reason it would be. And because we froze the encoder, the predictor has no way to *reshape* it. It is stuck decoding edit effects from a space that does not encode them in any accessible form.

So here is the tension, stated plainly:

> The same choice that makes the system efficient is the choice that makes it un-learnable. Freezing the encoder buys you cheap inference and immunity to collapse, and it takes away the one thing the model needed: the ability to become edit-predictable.

A model that trains the encoder jointly, the way most successful latent world models do, never fully hits this wall, because the encoder can adapt its geometry to whatever the predictor needs. By freezing, we opted out of that adaptation. We thought we were dodging the hard collapse problem. We were, but we walked straight into a different one. The collapse we avoided was the encoder collapsing onto a constant. The collapse we got was the predictor collapsing onto the identity. Phase-1 freezing prevents the first and quietly guarantees the second.

That is the lesson, and it is not specific to DNA. Any time you take a frozen foundation encoder and try to learn an action-conditioned dynamics model on top of it, you should ask: *is the effect of my action even visible in this frozen geometry, and is it big enough to dominate the do-nothing baseline?* If the answer to either is no, you have a residual trap waiting for you.

## A smaller, honest aside about the efficiency story

There is a second negative result that I have to report, because the whole point of this project is to report things honestly.

Remember the selling point, *pay for the encoder once, query it forever*. We never actually demonstrated it. The one latency number we measured, the released single-variant score path, came in at about **115 seconds per variant**. That sounds catastrophic, and it would be, except it is not measuring what it looks like. That number is a cold process: spin up Python, load a gigabyte of model weights from disk, run the encoder twice (once for the reference window, once for the edited one), and only then do the cheap part. The genuine compute is a fraction of a second. The rest is startup and loading.

The amortized regime, warm cache, model already resident, the predictor doing latent-only rollouts with the big model never running, is the regime the entire architecture is designed for. We did not measure it. The honest position is therefore not "GenoLeWM is slow" and not "GenoLeWM is fast." It is "the efficiency thesis is architecturally sound and operationally untested." I would rather say that than quietly drop a 115-second number into a footnote or pretend a cold-start measurement validates a caching argument it never touched.

The one place latent-only speed actually showed up, KV-cached rollout, gave a 2.41x speedup at a 5-edit horizon and 2.47x at a 20-edit horizon over the naive version, roughly flat in the number of steps, short of the 5x we were aiming for, and measured on toy dimensions rather than the real model. So even the good news is small and hedged. That is fine. Hedged and true beats clean and wrong.

## Why bother writing up a failure

Two reasons.

The first is the trap itself. "Predicting in latent space against a frozen encoder is hard when the action's effect is small relative to the embedding" is a transferable, falsifiable claim. It tells you something about a whole design pattern, not just one model. A clean negative with a mechanism is a real contribution. A messy negative with a shrug is not, and the difference is entirely in whether you understood *why*.

The second is reproducibility, and this is the part of the project I am most willing to defend. Every number in the benchmark is content-addressed. The model has an identity that is literally the hash of its manifest. Every scoring run emits a receipt binding the model id, the input, and the output. And the release pipeline does something I have not seen elsewhere: the tool that renders the results report **re-derives the report from the raw artifacts and refuses to pass unless the negative findings are present in them.** It hard-codes, as a check, that the non-coding AUROC gap is negative, that the rollout is below baseline, that the speed target was missed. You cannot quietly launder this project into a success without the verifier screaming. The infrastructure makes the honesty mechanical instead of optional.

## How you would actually find out if it is learnable

A negative result should leave a clear set of next experiments, ranked by how much they would teach you. The diagnosis above, *the effect of an edit is not legible in the frozen geometry*, suggests a few sharp ones.

**Probe before you build.** Before training any more predictors, just ask whether `Δ` is even linearly recoverable from the frozen encoder. Fit a linear probe and an MLP probe to predict the residual, and look at the gap. If a linear probe already does well, the signal is there and the predictor or the loss is at fault. If even the MLP fails, the encoder simply does not expose edit effects, and no amount of predictor tuning will save you. This is the cheapest, highest-information experiment in the whole program, and it should have come first.

**Unfreeze, carefully.** Let a lightweight adapter (LoRA) gently modify the encoder under an anti-collapse regularizer, so the geometry can bend toward making `Δ` extractable. This reintroduces the collapse risk we froze our way out of, which is the honest cost: you cannot have both the efficiency of freezing and the learnability of adaptation in the same phase. Pick your problem.

**Beat the right baselines, on the right scale.** Sixteen variants is not an evaluation. Size the slices to actually detect a two-point AUROC difference, and make the baselines adversarial: a linear map from `s_t` to `Δ`, a nearest-neighbor retrieval of `Δ` from similar contexts, the encoder's own likelihood difference. If the trained predictor cannot beat *interpolation*, it is not adding anything, and you want to know that early.

**Curriculum toward the visible edits.** Most single SNVs are nearly invisible to the encoder, which is exactly why the copy baseline is so strong. Start training on the edits that move the embedding the most: nonsense mutations, splice disruptions, edits in conserved regions. If the model cannot learn even the loud edits, the quiet ones are hopeless.

The unifying question, the one I think is genuinely open and worth someone's time, is this: *is edit-conditioned latent prediction learnable at all against a near-frozen genomic encoder, and if so, at what compute?* I do not know the answer. This project rules out the easiest version of "yes" and explains why. That is what it leaves behind.

## Takeaways

If you skim, take these.

A world model for genomic edits is a clean idea: freeze a DNA encoder, learn a tiny predictor from (state, edit) to next state, get cheap scoring, rollout, and planning for free. It is the kind of idea that should work.

It did not, and the reason is the latent-residual trap. A single edit barely moves a 1024-dimensional frozen embedding, so "predict no change" is a near-perfect baseline, and a residual predictor that starts at the identity is gradient-pulled into a worse-than-identity copy. The benchmark deficits are downstream of this one fact.

The deep reason is a tension you should carry to other problems: freezing an encoder gives you efficiency and collapse-immunity, and it takes away the encoder's ability to make your action's effect easy to read out. Those are the same decision. Frozen-feature world models live or die on whether the action is already visible in the frozen geometry.

And reproducibility is not paperwork. When your honest finding is negative, the infrastructure that makes the negative un-erasable is what lets anyone trust it.

I wanted this to work. It didn't. The reason it didn't is the part worth keeping.

---

*GenoLeWM is an alpha research system, not a clinical or diagnostic tool, and nothing here is medical evidence. Code, the released checkpoint, the benchmark artifacts, and a full technical write-up live in the [project repository](https://github.com/AbdelStark/GenoLeWM).*
