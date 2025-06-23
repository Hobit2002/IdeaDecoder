class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0][0];
    if (input) {
      this.port.postMessage([...input]); // flatten for easier consumption
    }
    return true;
  }
}

registerProcessor('recorder-processor', RecorderProcessor);
