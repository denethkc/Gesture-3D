// One Euro filter: low jitter when the hand is slow, low lag when it is fast.
export class OneEuro {
  constructor(minCutoff = 1.4, beta = 7, dCutoff = 1) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.x = null;
    this.dx = 0;
    this.t = null;
  }
  alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  filter(v, tMs) {
    if (this.x === null) {
      this.x = v;
      this.t = tMs;
      return v;
    }
    const dt = Math.max((tMs - this.t) / 1000, 1e-3);
    const dv = (v - this.x) / dt;
    this.dx += this.alpha(this.dCutoff, dt) * (dv - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += this.alpha(cutoff, dt) * (v - this.x);
    this.t = tMs;
    return this.x;
  }
  reset() {
    this.x = null;
    this.dx = 0;
    this.t = null;
  }
}

// Frame-rate independent easing. Call every render frame.
export const damp = (cur, target, lambda, dt) =>
  cur + (target - cur) * (1 - Math.exp(-lambda * dt));

export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const clamp01 = (v) => clamp(v, 0, 1);
