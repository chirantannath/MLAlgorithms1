
import java.lang.ref.SoftReference;
import java.util.*;
import java.util.function.*;
//import java.util.stream.*;

public class Perceptron {
  /**
   * Number of input layer nodes, i.e. the size of the input vector this system
   * accepts.
   * 
   * @implNote Actual number of inputs handled internally is <strong>one extra for
   *           bias values</strong>, where the extra input is fixed to {@code 1d}.
   */
  public final int inputLayerSize;
  /**
   * Number of output layer nodes, i.e. the size of the output vector this system
   * produces.
   */
  public final int outputLayerSize;
  /** Activation function. */
  public final ActivationFunction activationFunction;
  /**
   * Weight matrix. <strong>This is already in transpose form</strong>, i.e.
   * {@code weights[i][j]} represents the weight from the j<sup>th</sup> input to
   * the i<sup>th</sup> output. The matrix is of dimensions such that
   * <strong>there is an extra input fixed at {@code 1d}</strong> for bias values.
   */
  protected final double[][] weights;
  /**
   * Bit flags that are set to true if the corresponding weight is fixed and is
   * not to be changed.
   */
  protected final BitSet[] fixedWeightsMask;
  /** Stored inputs. */
  protected final double[] inputs;
  /**
   * Stored net inputs to output layer <strong>before application of activation
   * function</strong>.
   */
  protected final double[] netInputs;
  /** Stored outputs. */
  protected final double[] outputs;
  /** Delta matrix to change weights. Cache this to prevent reallocations. */
  private SoftReference<double[][]> deltaCache = null;
  /**
   * Derivatives of the loss function with respect to net input values to output
   * layer. Cache this to prevent reallocations.
   */
  private SoftReference<double[]> betaCache = null;

  public Perceptron(
      int inputLayerSize,
      int outputLayerSize,
      ActivationFunction activationFunction,
      DoubleSupplier initialWeightsGenerator) {
    if (inputLayerSize <= 0)
      throw new IllegalArgumentException("inputLayerSize <= 0");
    this.inputLayerSize = inputLayerSize;
    if (outputLayerSize <= 0)
      throw new IllegalArgumentException("outputLayerSize <= 0");
    this.outputLayerSize = outputLayerSize;
    this.activationFunction = Objects.requireNonNull(activationFunction, "activationFunction");

    // inputLayerSize + 1 for activation function
    weights = new double[outputLayerSize][inputLayerSize + 1];
    fixedWeightsMask = new BitSet[outputLayerSize];
    inputs = new double[inputLayerSize + 1];
    inputs[inputLayerSize] = 1d;
    netInputs = new double[outputLayerSize];
    outputs = new double[outputLayerSize];

    for (int r = 0; r < weights.length; r++) {
      fixedWeightsMask[r] = new BitSet(inputLayerSize + 1);
      weights[r][inputLayerSize] = 0; // bias values usually set to 0
      if (initialWeightsGenerator == null)
        continue;
      for (int c = 0; c < inputLayerSize; c++)
        weights[r][c] = initialWeightsGenerator.getAsDouble();
    }
  }

  public Perceptron(int inputLayerSize, int outputLayerSize, ActivationFunction activationFunction,
      Random weightsSource) {
    this(inputLayerSize, outputLayerSize, activationFunction, weightsSource::nextGaussian);
  }

  public Perceptron(int inputLayerSize, int outputLayerSize, ActivationFunction activationFunction) {
    this(inputLayerSize, outputLayerSize, activationFunction, new Random());
  }

  public Perceptron(int inputLayerSize, int outputLayerSize) {
    this(inputLayerSize, outputLayerSize, ActivationFunction.SIGMOID);
  }

  public Perceptron(Perceptron other) {
    this(other.inputLayerSize, other.outputLayerSize, other.activationFunction, (DoubleSupplier) null);
    for (int wr = 0; wr < weights.length; wr++) {
      final var row = other.weights[wr];
      System.arraycopy(row, 0, weights[wr], 0, row.length);
      fixedWeightsMask[wr].clear();
      fixedWeightsMask[wr].or(other.fixedWeightsMask[wr]);
    }
    System.arraycopy(other.inputs, 0, inputs, 0, inputs.length);
    System.arraycopy(other.netInputs, 0, netInputs, 0, netInputs.length);
    System.arraycopy(other.outputs, 0, outputs, 0, outputs.length);
  }

  public int getWeightMatrixNumRows() {
    return outputLayerSize;
  }

  public int getWeightMatrixNumColumns() {
    return inputLayerSize + 1;
  }

  public double[][] getWeights() {
    return Utils.matrixClone(weights);
  }

  public double getWeight(int outputIndex, int inputIndex) {
    return weights[outputIndex][inputIndex];
  }

  public void setWeight(int outputIndex, int inputIndex, double value) {
    weights[outputIndex][inputIndex] = value;
  }

  public boolean isWeightFixed(int outputIndex, int inputIndex) {
    return fixedWeightsMask[outputIndex].get(inputIndex);
  }

  public void setWeightFixed(int outputIndex, int inputIndex, boolean flag) {
    fixedWeightsMask[outputIndex].set(inputIndex, flag);
  }

  public double[] getInputs() {
    return Arrays.copyOf(inputs, inputLayerSize);
  }

  public void setInputs(double[] inputs) {
    if (inputs.length != inputLayerSize)
      throw new IllegalArgumentException();
    System.arraycopy(inputs, 0, this.inputs, 0, inputLayerSize);
  }

  public double getInput(int index) {
    Objects.checkIndex(index, inputLayerSize);
    return inputs[index];
  }

  public void setInput(int index, double value) {
    Objects.checkIndex(index, inputLayerSize);
    inputs[index] = value;
  }

  public double[] getOutputs() {
    return Arrays.copyOf(outputs, outputLayerSize);
  }

  public double getOutput(int index) {
    return outputs[index];
  }

  public void processInput() {
    inputs[inputLayerSize] = 1d;
    Utils.matrixMultiply(weights, inputs, netInputs);
    for (int i = 0; i < outputLayerSize; i++)
      outputs[i] = activationFunction.applyAsDouble(netInputs[i]);
  }

  protected double[][] getWeightDeltas() {
    double[][] deltas = (deltaCache == null) ? null : deltaCache.get();
    if (deltas == null) {
      deltas = new double[outputLayerSize][inputLayerSize + 1];
      deltaCache = new SoftReference<>(deltas);
    }
    return deltas;
  }

  protected double[] getBetas() {
    double[] betas = (betaCache == null) ? null : betaCache.get();
    if (betas == null) {
      betas = new double[outputs.length];
      betaCache = new SoftReference<>(betas);
    }
    return betas;
  }

  protected void fillBetas(double[] expectedValues, LossFunction lossFunction, double[] betas) {
    if (outputLayerSize != expectedValues.length)
      throw new IllegalArgumentException();

    for (int outputIndex = 0; outputIndex < outputs.length; outputIndex++)
      betas[outputIndex] = lossFunction.applyDerivativeWRTActual(expectedValues, outputs, outputIndex)
          * activationFunction.applyDerivative(netInputs[outputIndex]);
  }

  protected double fillWeightDeltas(double[] betas, double learningRate,
      double[][] deltas) {

    double changeMax = 0, change;
    for (int inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
      for (int outputIndex = 0; outputIndex < outputs.length; outputIndex++) {
        deltas[outputIndex][inputIndex] = change = isWeightFixed(outputIndex, inputIndex) ? 0d
            : -learningRate * betas[outputIndex] * inputs[inputIndex];
        changeMax = Math.max(changeMax, Math.abs(change));
      }
    }

    return changeMax;
  }

  public final double adjustWeights(double[] expectedValues, LossFunction lossFunction, double learningRate) {
    if (outputLayerSize != expectedValues.length)
      throw new IllegalArgumentException("expectedValues");

    final double[][] deltas = getWeightDeltas();
    final double[] betas = getBetas();
    fillBetas(expectedValues, lossFunction, betas);
    double changeMax = fillWeightDeltas(betas, learningRate, deltas);
    Utils.matrixAddAccumulate(weights, deltas);
    return changeMax;
  }
}
