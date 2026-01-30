/**
 * MoshiProcessor - AudioWorklet for low-latency streaming audio playback
 * Ported from the original PersonaPlex client implementation.
 */

function asMs(samples, sr) {
  return (samples * 1000 / sr).toFixed(1);
}

function asSamples(mili, sr) {
  return Math.round(mili * sr / 1000);
}

class MoshiProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Buffer length definitions based on 80ms frames
    const frameSize = asSamples(80, sampleRate);

    // We wait to have at least that many samples before starting to play
    this.initialBufferSamples = 1 * frameSize;

    // Once we have enough samples, we further wait that long before starting to play
    this.partialBufferSamples = asSamples(10, sampleRate);

    // If the buffer length goes over that many, we will drop the oldest packets
    this.maxBufferSamples = asSamples(10, sampleRate);

    // Increments for adaptive buffering
    this.partialBufferIncrement = asSamples(5, sampleRate);
    this.maxPartialWithIncrements = asSamples(80, sampleRate);
    this.maxBufferSamplesIncrement = asSamples(5, sampleRate);
    this.maxMaxBufferWithIncrements = asSamples(80, sampleRate);

    this.initState();

    this.port.onmessage = (event) => {
      if (event.data.type === "reset") {
        this.initState();
        return;
      }

      const frame = event.data.frame;
      this.frames.push(frame);

      if (this.currentSamples() >= this.initialBufferSamples && !this.started) {
        this.start();
      }

      // Drop packets if buffer is too full
      if (this.currentSamples() >= this.totalMaxBufferSamples()) {
        const target = this.initialBufferSamples + this.partialBufferSamples;
        while (this.currentSamples() > target) {
          const first = this.frames[0];
          let toRemove = this.currentSamples() - target;
          toRemove = Math.min(first.length - this.offsetInFirstBuffer, toRemove);
          this.offsetInFirstBuffer += toRemove;
          this.timeInStream += toRemove / sampleRate;
          if (this.offsetInFirstBuffer === first.length) {
            this.frames.shift();
            this.offsetInFirstBuffer = 0;
          }
        }
        this.maxBufferSamples += this.maxBufferSamplesIncrement;
        this.maxBufferSamples = Math.min(this.maxMaxBufferWithIncrements, this.maxBufferSamples);
      }

      // Report stats back to main thread
      this.port.postMessage({
        totalAudioPlayed: this.totalAudioPlayed,
        actualAudioPlayed: this.actualAudioPlayed,
        delay: event.data.micDuration - this.timeInStream,
        minDelay: this.minDelay,
        maxDelay: this.maxDelay,
      });
    };
  }

  initState() {
    this.frames = [];
    this.offsetInFirstBuffer = 0;
    this.firstOut = false;
    this.remainingPartialBufferSamples = 0;
    this.timeInStream = 0;
    this.resetStart();

    // Metrics
    this.totalAudioPlayed = 0;
    this.actualAudioPlayed = 0;
    this.maxDelay = 0;
    this.minDelay = 2000;

    // Reset buffer params
    this.partialBufferSamples = asSamples(10, sampleRate);
    this.maxBufferSamples = asSamples(10, sampleRate);
  }

  totalMaxBufferSamples() {
    return this.maxBufferSamples + this.partialBufferSamples + this.initialBufferSamples;
  }

  currentSamples() {
    let samples = 0;
    for (let k = 0; k < this.frames.length; k++) {
      samples += this.frames[k].length;
    }
    samples -= this.offsetInFirstBuffer;
    return samples;
  }

  resetStart() {
    this.started = false;
  }

  start() {
    this.started = true;
    this.remainingPartialBufferSamples = this.partialBufferSamples;
    this.firstOut = true;
  }

  canPlay() {
    return this.started && this.frames.length > 0 && this.remainingPartialBufferSamples <= 0;
  }

  process(inputs, outputs, parameters) {
    const delay = this.currentSamples() / sampleRate;
    if (this.canPlay()) {
      this.maxDelay = Math.max(this.maxDelay, delay);
      this.minDelay = Math.min(this.minDelay, delay);
    }

    const output = outputs[0][0];

    if (!this.canPlay()) {
      if (this.actualAudioPlayed > 0) {
        this.totalAudioPlayed += output.length / sampleRate;
      }
      this.remainingPartialBufferSamples -= output.length;
      return true;
    }

    if (this.firstOut) {
      // First output after buffer fill
    }

    let outIdx = 0;
    while (outIdx < output.length && this.frames.length) {
      const first = this.frames[0];
      const toCopy = Math.min(first.length - this.offsetInFirstBuffer, output.length - outIdx);
      output.set(first.subarray(this.offsetInFirstBuffer, this.offsetInFirstBuffer + toCopy), outIdx);
      this.offsetInFirstBuffer += toCopy;
      outIdx += toCopy;
      if (this.offsetInFirstBuffer === first.length) {
        this.offsetInFirstBuffer = 0;
        this.frames.shift();
      }
    }

    if (this.firstOut) {
      this.firstOut = false;
      // Fade in
      for (let i = 0; i < outIdx; i++) {
        output[i] *= i / outIdx;
      }
    }

    if (outIdx < output.length) {
      // Ran out of buffer - increase partial buffer
      this.partialBufferSamples += this.partialBufferIncrement;
      this.partialBufferSamples = Math.min(this.partialBufferSamples, this.maxPartialWithIncrements);
      this.resetStart();
      // Fade out
      for (let i = 0; i < outIdx; i++) {
        output[i] *= (outIdx - i) / outIdx;
      }
    }

    this.totalAudioPlayed += output.length / sampleRate;
    this.actualAudioPlayed += outIdx / sampleRate;
    this.timeInStream += outIdx / sampleRate;
    return true;
  }
}

registerProcessor("moshi-processor", MoshiProcessor);
