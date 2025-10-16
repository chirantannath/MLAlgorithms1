
import java.lang.ref.SoftReference;
import java.util.*;
import java.util.function.*;

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
  protected final double[][] layerNetInputs;
  /**
   * Stored outputs of each layer in a pass.
   */
  protected final double[][] layerOutputs;
  /** Delta matrix to change weights. Cache this to prevent reallocations. */
  private SoftReference<double[][][]> deltaCache = null;
  /**
   * Stored reference of the derivative of the loss function with respect to the
   * output at each layer, multiplied by the derivative of that output with
   * respect to the net input of the node in that layer.
   * 
   * <p>We call it the "beta" value here, but do note that most textbooks refer to
   * this as the <i>delta</i> values. We are using the "delta" name for something
   * else (actual change in weight).</p>
   */
  private SoftReference<double[][]> betaCache = null;

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
    layerNetInputs = new double[hiddenLayerSizes.length + 1][];
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
      layerNetInputs[layer] = new double[layerSize];
      // Number of outputs in this layer == number of nodes, plus one for bias to next
      // layer
      layerOutputs[layer + 1] = new double[layerSize + 1];
      layerOutputs[layer + 1][layerSize] = 1d;
      // Obviously at the output layer, the last fixed output is unused.

      // For weight matrix, number of columns is number of outputs of previous layer
      // (with bias implicitly included)
      final int wclen = layerOutputs[layer].length;
      // Number of rows is number of inputs to this layer
      final int wrlen = layerNetInputs[layer].length;
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

  public MultiLayerPerceptron(
      int inputLayerSize,
      int[] hiddenLayerSizes,
      int outputLayerSize,
      IntFunction<? extends ActivationFunction> activationFunctions,
      Random weightsSource) {
    this(inputLayerSize, hiddenLayerSizes, outputLayerSize, activationFunctions, weightsSource::nextGaussian);
  }

  public MultiLayerPerceptron(
      int inputLayerSize,
      int[] hiddenLayerSizes,
      int outputLayerSize,
      IntFunction<? extends ActivationFunction> activationFunctions) {
    this(inputLayerSize, hiddenLayerSizes, outputLayerSize, activationFunctions, new Random()::nextGaussian);
  }

  public MultiLayerPerceptron(int inputLayerSize, int[] hiddenLayerSizes, int outputLayerSize,
      ActivationFunction activationFunction) {
    this(inputLayerSize, hiddenLayerSizes, outputLayerSize, layer -> activationFunction);
  }

  public MultiLayerPerceptron(MultiLayerPerceptron other) {
    this(other.inputLayerSize, other.hiddenLayerSizes, other.outputLayerSize, other.activationFunctions, () -> 0d);
    System.arraycopy(other.layerOutputs[0], 0, layerOutputs[0], 0, inputLayerSize + 1);
    for (int layer = 0; layer <= hiddenLayerSizes.length; layer++) {
      // final int layerSize = layer < hiddenLayerSizes.length ?
      // hiddenLayerSizes[layer] : outputLayerSize;

      // System.arraycopy(other.layerNetInputs[layer], 0, layerNetInputs[layer], 0,
      // layerSize);
      // System.arraycopy(other.layerOutputs[layer + 1], 0, layerOutputs[layer + 1],
      // 0, layerSize + 1);
      final int wclen = layerOutputs[layer].length; // Bias implicitly included from previous layer
      final int wrlen = layerNetInputs[layer].length;

      for (int r = 0; r < wrlen; r++) {
        fixedWeightsMask[layer][r].clear();
        fixedWeightsMask[layer][r].or(other.fixedWeightsMask[layer][r]);
        System.arraycopy(other.weights[layer][r], 0, weights[layer][r], 0, wclen);
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
  public void processInput() {
    layerOutputs[0][inputLayerSize] = 1d; // fix bias multiplier
    for (int l = 0; l < weights.length; l++) {
      final int layer = l;
      final int layerSize = layer < hiddenLayerSizes.length ? hiddenLayerSizes[layer] : outputLayerSize;

      // Prepare input for this layer
      Utils.matrixMultiply(
          weights[layer], // weights for the INPUTS to this layer.
          layerOutputs[layer], // Output from previous layer
          layerNetInputs[layer] // Inputs for this layer
      );

      final var layerActivationFunction = activationFunctions.apply(layer);
      // Run activation function
      for (int i = 0; i < layerSize; i++)
        layerOutputs[layer + 1][i] = layerActivationFunction.applyAsDouble(layerNetInputs[layer][i]);
      layerOutputs[layer + 1][layerSize] = 1d; // fix bias multiplier
    }
  }

  protected double[][][] getWeightDeltas() {
    var deltas = deltaCache == null ? null : deltaCache.get();
    if (deltas == null) {
      deltas = new double[hiddenLayerSizes.length + 1][][];
      for (int layer = 0; layer <= hiddenLayerSizes.length; layer++) {
        final int wclen = layerOutputs[layer].length;
        final int wrlen = layerNetInputs[layer].length;
        deltas[layer] = new double[wrlen][wclen];
      }
      deltaCache = new SoftReference<>(deltas);
    }
    return deltas;
  }

  protected double[][] getBetas() {
    var betas = betaCache == null ? null : betaCache.get();
    if (betas == null) {
      betas = new double[hiddenLayerSizes.length + 1][];
      for (int layer = 0; layer <= hiddenLayerSizes.length; layer++) {
        final int layerSize = layer < hiddenLayerSizes.length ? hiddenLayerSizes[layer] : outputLayerSize;
        betas[layer] = new double[layerSize];
      }
      betaCache = new SoftReference<>(betas);
    }
    return betas;
  }

  protected void fillBetas(double[] expectedValues, LossFunction lossFunction, double[][] betas) {
    if (outputLayerSize != expectedValues.length)
      throw new IllegalArgumentException();

    // First, compute for the last layer
    int l = hiddenLayerSizes.length; // last layer
    {
      final var outputActivationFunction = activationFunctions.apply(l);
      for (int i = 0; i < outputLayerSize; i++)
        betas[l][i] = lossFunction.applyDerivativeWRTActual(expectedValues, layerOutputs[l + 1], i)
            * outputActivationFunction.applyDerivative(layerNetInputs[l][i]);
    }
    l--;
    // Then for the hidden layers, computing backward
    for (; l >= 0; l--) {
      final int layerSize = hiddenLayerSizes[l];
      final int nextLayerSize = l < (hiddenLayerSizes.length - 1) ? hiddenLayerSizes[l + 1] : outputLayerSize;
      final var layerActivationFunction = activationFunctions.apply(l);
      for (int i = 0; i < layerSize; i++) {
        betas[l][i] = 0;
        for (int p = 0; p < nextLayerSize; p++)
          betas[l][i] += betas[l + 1][p] * weights[l + 1][p][i];
        betas[l][i] *= layerActivationFunction.applyDerivative(layerNetInputs[l][i]);
      }
    }
  }

  protected double fillWeightDeltas(double learningRate, double[][] betas, double[][][] deltas) {
    double changeMax = 0;
    for (int layer = 0; layer <= hiddenLayerSizes.length; layer++) {
      int prevLayerSize = layerOutputs[layer].length;
      int layerSize = betas[layer].length;
      for (int i = 0; i < layerSize; i++)
        for (int j = 0; j < prevLayerSize; j++) {
          double change = deltas[layer][i][j] = isWeightFixed(layer, i, j) ? 0d
              : -learningRate * betas[layer][i] * layerOutputs[layer][j];
          changeMax = Math.max(changeMax, Math.abs(change));
        }
    }
    return changeMax;
  }

  /**
   * Do backward pass to adjust weights.
   * 
   * @implNote This version applies the standard backpropagation algorithm.
   */
  public final double adjustWeights(double[] expectedValues, LossFunction lossFunction, double learningRate) {
    Objects.requireNonNull(lossFunction);
    if (outputLayerSize != expectedValues.length)
      throw new IllegalArgumentException();

    final double[][][] deltas;
    double changeMax;
    {
      final var betas = getBetas();
      fillBetas(expectedValues, lossFunction, betas);
      deltas = getWeightDeltas();
      changeMax = fillWeightDeltas(learningRate, betas, deltas);
    }

    for (int layer = 0; layer < weights.length; layer++)
      Utils.matrixAddAccumulate(weights[layer], deltas[layer]);

    return changeMax;
  }
}
