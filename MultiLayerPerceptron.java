
import java.util.*;
import java.util.function.*;
import java.util.stream.*;

public class MultiLayerPerceptron {
  /**
   * Number of input layer nodes, i.e. the size of the input vector this system
   * accepts.
   * 
   * @implNote Actual number of inputs handled internally is <strong>one extra for
   *           bias values</strong>, where the extra input is fixed to {@code 1d}.
   */
  public final int inputLayerSize;
  /**
   * Number of hidden layer nodes per layer.
   */
  protected final int[] hiddenLayerSizes;
  /**
   * Number of output layer nodes, i.e. the size of the output vector this system
   * produces.
   */
  public final int outputLayerSize;
  /**
   * Activation function <strong>per layer</strong>, including the output layer
   * (but excluding the input layer).
   */
  public final IntFunction<? extends ActivationFunction> activationFunctions;
  /**
   * Weight matrices. They are of dimensions such that <strong>every layer has an
   * extra input of 1</strong>.
   */
  protected final double[][][] weights;
  /**
   * Bit flags that are set to true if the corresponding weight is fixed and is
   * not to be changed.
   */
  protected final BitSet[][] fixedWeightsMask;
  /**
   * Stored <strong>net total</strong> inputs of each hidden and output layer in a
   * pass.
   */
  protected final double[][] layerInputs;
  /**
   * Stored outputs of each layer in a pass.
   */
  protected final double[][] layerOutputs;

  public MultiLayerPerceptron(
      int inputLayerSize,
      int[] hiddenLayerSizes,
      int outputLayerSize,
      IntFunction<? extends ActivationFunction> activationFunctions,
      DoubleSupplier initialWeightsGenerator) {
    if (inputLayerSize <= 0)
      throw new IllegalArgumentException("inputLayerSize <= 0");
    this.inputLayerSize = inputLayerSize;
    this.hiddenLayerSizes = Objects.requireNonNullElse(hiddenLayerSizes, new int[0]);
    if (outputLayerSize <= 0)
      throw new IllegalArgumentException("outputLayerSize <= 0");
    this.outputLayerSize = outputLayerSize;
    this.activationFunctions = Objects.requireNonNull(activationFunctions, "activationFunctions");
    initialWeightsGenerator = Objects.requireNonNullElse(initialWeightsGenerator, new Random()::nextGaussian);

    // Now initialize internal arrays
    weights = new double[hiddenLayerSizes.length + 1][][];
    fixedWeightsMask = new BitSet[hiddenLayerSizes.length + 1][];
    layerInputs = new double[hiddenLayerSizes.length + 1][];
    layerOutputs = new double[hiddenLayerSizes.length + 2][];

    // For the very first input layer
    // No weights, no inputs. input layer does have an output tho.
    layerOutputs[0] = new double[inputLayerSize + 1]; // 1 extra for bias
    layerOutputs[0][inputLayerSize] = 1d; // bias input fixed at 1.

    // For every hidden layer and the output layer
    for (int layer = 0; layer <= hiddenLayerSizes.length; layer++) {
      final int layerSize = layer < hiddenLayerSizes.length ? hiddenLayerSizes[layer] : outputLayerSize;
      if (layerSize <= 0)
        throw new IllegalArgumentException(String.format("hiddenLayerSizes[%d] <= 0", layer));

      // Number of inputs in this layer == number of nodes
      layerInputs[layer] = new double[layerSize];
      // Number of outputs in this layer == number of nodes, plus one for bias to next
      // layer
      layerOutputs[layer + 1] = new double[layerSize + 1];
      layerOutputs[layer + 1][layerSize] = 1d;
      // Obviously at the output layer, the last fixed output is unused.

      // For weight matrix, number of columns is number of outputs of previous layer
      // (with bias implicitly included)
      final int wclen = layerOutputs[layer].length;
      // Number of rows is number of inputs to this layer
      final int wrlen = layerInputs[layer].length;
      weights[layer] = new double[wrlen][wclen];
      fixedWeightsMask[layer] = new BitSet[wrlen];
      for (int r = 0; r < wrlen; r++) {
        fixedWeightsMask[layer][r] = new BitSet(wclen);
        for (int c = 0; c < (wclen - 1); c++)
          // Actual weights usually from a distribution
          weights[layer][r][c] = initialWeightsGenerator.getAsDouble();
        // Initial biases usually 0
        weights[layer][r][wclen - 1] = 0d;
      }
    }
  }

  public int getNumHiddenLayers() {
    return hiddenLayerSizes.length;
  }

  public int getHiddenLayerSize(int hiddenLayerIndex) {
    return hiddenLayerSizes[hiddenLayerIndex];
  }

  public int getWeightMatrixNumRows(int layerIndex) {
    return weights[layerIndex].length;
  }

  public int getWeightMatrixNumColumns(int layerIndex) {
    return weights[layerIndex][0].length;
  }

  public double[][] getWeight(int layerIndex) {
    return Utils.matrixClone(weights[layerIndex]);
  }

  public double getWeight(int layerIndex, int layerNodeIndex, int prevLayerNodeIndex) {
    return weights[layerIndex][layerNodeIndex][prevLayerNodeIndex];
  }

  public void setWeight(int layerIndex, int layerNodeIndex, int prevLayerNodeIndex, double value) {
    weights[layerIndex][layerNodeIndex][prevLayerNodeIndex] = value;
  }

  public boolean isWeightFixed(int layerIndex, int layerNodeIndex, int prevLayerNodeIndex) {
    return fixedWeightsMask[layerIndex][layerNodeIndex].get(prevLayerNodeIndex);
  }

  public void setWeightFixed(int layerIndex, int layerNodeIndex, int prevLayerNodeIndex, boolean flag) {
    fixedWeightsMask[layerIndex][layerNodeIndex].set(prevLayerNodeIndex, flag);
  }

  public double[] getNetworkInputs() {
    return Arrays.copyOf(layerOutputs[0], inputLayerSize);
  }

  public void setNetworkInputs(double[] inputs) {
    if (inputs.length != inputLayerSize)
      throw new IllegalArgumentException();
    System.arraycopy(inputs, 0, layerOutputs[0], 0, inputLayerSize);
  }

  public double getNetworkInput(int index) {
    Objects.checkIndex(index, inputLayerSize);
    return layerOutputs[0][index];
  }

  public void setNetworkInput(int index, double value) {
    Objects.checkIndex(index, inputLayerSize);
    layerOutputs[0][index] = value;
  }

  public double[] getNetworkOutputs() {
    return Arrays.copyOf(layerOutputs[layerOutputs.length - 1], outputLayerSize);
  }

  public double getNetworkOutput(int index) {
    Objects.checkIndex(index, outputLayerSize);
    return layerOutputs[layerOutputs.length - 1][index];
  }

  /** Execute one forward pass through the network. */
  public void forwardPass() {
    for (int l = 0; l < weights.length; l++) {
      final int layer = l;
      final int layerSize = layer < hiddenLayerSizes.length ? hiddenLayerSizes[layer] : outputLayerSize;

      // Prepare input for this layer
      Utils.matrixMultiply(
          weights[layer], // weights for the INPUTS to this layer.
          layerOutputs[layer], // Output from previous layer
          layerInputs[layer] // Inputs for this layer
      );

      // Run activation function
      IntStream.range(0, layerSize)//.unordered().parallel()
          .forEach(
              i -> layerOutputs[layer + 1][i] = activationFunctions.apply(layer).applyAsDouble(layerInputs[layer][i]));
    }
  }

  /** 
   * Do backward pass to adjust weights. 
   * @implNote This version applies the standard backpropagation algorithm.
   */
  public void backwardPass(double[] expectedOutputs, LossFunction lossFunction, double learningRate) {
    Objects.requireNonNull(lossFunction);
    if (outputLayerSize != expectedOutputs.length)
      throw new IllegalArgumentException();
    //TODO: Complete this!
    throw new UnsupportedOperationException();
  }
}
