import java.lang.ref.SoftReference;
import java.util.*;
import java.util.function.*;

public class MLPClassifier<C> implements Classifier<Float64Row, C> {
  protected MultiLayerPerceptron mainPerceptron = null;

  public final int rowLength;

  protected int[] hiddenLayerSizes;

  public final double learningRate;

  public final long maxEpochs;

  public final double maxDeltaThreshold;

  public final ActivationFunction hiddenActivationFunction;

  public final ActivationFunction outputActivationFunction;

  public final LossFunction lossFunction;

  public final DoubleSupplier initialWeightsGenerator;

  protected List<Pair<Float64Row, C>> data = new ArrayList<>();
  // /**
  // * Held-out set prepared for validation of loss function to prevent
  // overfitting.
  // * Currently sets 10% for use with a minimum of "number of classes seen" + 1.
  // */
  // protected List<Pair<Float64Row, C>> validationData;

  protected Map<C, Integer> classesSeen = new LinkedHashMap<>();

  protected List<C> classesSeenList = new ArrayList<>();

  protected final ThreadLocal<SoftReference<MultiLayerPerceptron>> threadLocalPerceptrons = new ThreadLocal<>();

  protected BiConsumer<? super Long, ? super Double> epochLossNotifier = null;

  public MLPClassifier(
      int rowLength,
      int[] hiddenLayerSizes,
      double learningRate,
      long maxEpochs,
      double maxDeltaThreshold,
      ActivationFunction hiddenActivationFunction,
      ActivationFunction outputActivationFunction,
      LossFunction lossFunction,
      DoubleSupplier initialWeightsGenerator) {
    this.hiddenActivationFunction = Objects.requireNonNull(hiddenActivationFunction, "hiddenActivationFunction");
    this.outputActivationFunction = Objects.requireNonNull(outputActivationFunction, "outputActivationFunction");
    this.lossFunction = Objects.requireNonNull(lossFunction, "lossFunction");
    this.initialWeightsGenerator = initialWeightsGenerator == null ? new Random()::nextGaussian
        : initialWeightsGenerator;

    Objects.requireNonNull(hiddenLayerSizes, "hiddenLayerSizes");
    for (int i = 0; i < hiddenLayerSizes.length; i++)
      if (hiddenLayerSizes[i] <= 0)
        throw new IllegalArgumentException(String.format("hiddenLayerSizes[%d] <= 0", i));
    this.hiddenLayerSizes = Arrays.copyOf(hiddenLayerSizes, hiddenLayerSizes.length);

    if (rowLength <= 0)
      throw new IllegalArgumentException("rowLength");
    this.rowLength = rowLength;
    if (learningRate <= 0)
      throw new IllegalArgumentException("learningRate");
    this.learningRate = learningRate;
    this.maxEpochs = maxEpochs;
    this.maxDeltaThreshold = maxDeltaThreshold;
  }

  public MLPClassifier(
      int rowLength,
      int[] hiddenLayerSizes,
      double learningRate,
      long maxEpochs,
      double maxDeltaThreshold,
      ActivationFunction hiddenActivationFunction,
      ActivationFunction outputActivationFunction,
      LossFunction lossFunction,
      Random initialWeightsGenerator) {
    this(rowLength, hiddenLayerSizes, learningRate, maxEpochs, maxDeltaThreshold, hiddenActivationFunction,
        outputActivationFunction, lossFunction, initialWeightsGenerator::nextGaussian);
  }

  public MLPClassifier(
      int rowLength,
      int[] hiddenLayerSizes,
      double learningRate,
      long maxEpochs,
      double maxDeltaThreshold,
      ActivationFunction hiddenActivationFunction,
      ActivationFunction outputActivationFunction,
      LossFunction lossFunction) {
    this(rowLength, hiddenLayerSizes, learningRate, maxEpochs, maxDeltaThreshold, hiddenActivationFunction,
        outputActivationFunction, lossFunction, new Random()::nextGaussian);
  }

  public MLPClassifier(
      int rowLength,
      int[] hiddenLayerSizes,
      double learningRate,
      long maxEpochs,
      double maxDeltaThreshold,
      ActivationFunction hiddenActivationFunction,
      ActivationFunction outputActivationFunction) {
    this(rowLength, hiddenLayerSizes, learningRate, maxEpochs, maxDeltaThreshold, hiddenActivationFunction,
        outputActivationFunction, LossFunction.SOFTMAX_LOG_LOSS);
  }

  public MLPClassifier(int rowLength, int[] hiddenLayerSizes, double learningRate, long maxEpochs,
      double maxDeltaThreshold, ActivationFunction hiddenActivationFunction) {
    this(rowLength, hiddenLayerSizes, learningRate, maxEpochs, maxDeltaThreshold, hiddenActivationFunction,
        ActivationFunction.SIGMOID, LossFunction.SOFTMAX_LOG_LOSS);
  }

  public MLPClassifier(int rowLength, int[] hiddenLayerSizes, double learningRate, long maxEpochs,
      double maxDeltaThreshold) {
    this(rowLength, hiddenLayerSizes, learningRate, maxEpochs, maxDeltaThreshold,
        ActivationFunction.SIGMOID);
  }

  public MLPClassifier(int rowLength, int[] hiddenLayerSizes, double learningRate, long maxEpochs) {
    this(rowLength, hiddenLayerSizes, learningRate, maxEpochs, 0d);
  }

  public MLPClassifier(int rowLength, int[] hiddenLayerSizes, double learningRate) {
    this(rowLength, hiddenLayerSizes, learningRate, Short.MAX_VALUE);
  }

  public MLPClassifier(int rowLength, double learningRate) {
    this(rowLength, new int[] { 100 }, learningRate);
  }

  @Override
  public void fit(Float64Row input, C trueOutput) {
    if (input.getRowLength() != rowLength)
      throw new IllegalArgumentException();
    data.add(new Pair<>(input, trueOutput));
    if (!classesSeen.containsKey(trueOutput)) {
      int newIndex = classesSeenList.size();
      classesSeenList.add(newIndex, trueOutput);
      classesSeen.put(trueOutput, newIndex);
    }
  }

  public synchronized void setEpochLossNotifier(BiConsumer<? super Long, ? super Double> notifier) {
    epochLossNotifier = notifier;
  }

  /**
   * Gets the number of <i>processing</i> layers (i.e. excluding the input layer).
   */
  public final int getNumProcessingLayers() {
    return hiddenLayerSizes.length + 1;
  }

  public ActivationFunction getProcessingLayerActivationFunction(int layerIndex) {
    if (layerIndex == hiddenLayerSizes.length)
      return outputActivationFunction;
    else if (layerIndex >= 0 && layerIndex < hiddenLayerSizes.length)
      return hiddenActivationFunction;
    else
      throw new IndexOutOfBoundsException();
  }

  @Override
  public synchronized void finishFitting() {
    if (data instanceof ArrayList<?> al)
      al.trimToSize();

    // Prepare held-out set
    final int numClasses = classesSeenList.size();
    // {
    // final int dataSize = data.size();
    // int heldOutLength = Math.max(Math.round(data.size() * 0.1F), numClasses + 1);
    // // This is a very bad solution, but try this anyway.
    // var subList = data.subList(dataSize - heldOutLength, dataSize);
    // var vd = new ArrayList<>(subList);
    // vd.trimToSize();
    // validationData = Collections.unmodifiableList(vd);
    // subList.clear();
    // assert data.size() < dataSize;
    // }

    data = Collections.unmodifiableList(data);
    classesSeen = Collections.unmodifiableMap(classesSeen);
    if (classesSeenList instanceof ArrayList<?> al)
      al.trimToSize();
    classesSeenList = Collections.unmodifiableList(classesSeenList);

    mainPerceptron = new MultiLayerPerceptron(
        rowLength,
        hiddenLayerSizes,
        numClasses,
        this::getProcessingLayerActivationFunction,
        initialWeightsGenerator);

    double[] expectedValues = new double[mainPerceptron.outputLayerSize];
    double[] actualValues = new double[mainPerceptron.outputLayerSize];
    Arrays.fill(expectedValues, 0d);
    //final var trainingLossFunction = LossFunction.scaledLossFunction(lossFunction, 1d / data.size());
    //double oldTotalLoss = -1;
    for (long epoch = 0; epoch < maxEpochs; epoch++) {
      double epochMaxDelta = 0d; // don't remove this default
      double totalLoss = 0d;

      for (var input : data) {
        var row = input.first();
        int clsIndex = classesSeen.get(input.second());
        for (int i = 0; i < rowLength; i++)
          mainPerceptron.setNetworkInput(i, row.getAsDouble(i));
        mainPerceptron.processInput();

        expectedValues[clsIndex] = 1d;

        if (epochLossNotifier != null) {
          for (int i = 0; i < mainPerceptron.outputLayerSize; i++)
            actualValues[i] = mainPerceptron.getNetworkOutput(i);
          totalLoss += lossFunction.applyAsDouble(expectedValues, actualValues);
        }

        double maxDelta = mainPerceptron.adjustWeights(expectedValues, lossFunction,
            learningRate);
        expectedValues[clsIndex] = 0d;
        epochMaxDelta = Math.max(epochMaxDelta, maxDelta);
      }

      if (epochLossNotifier != null)
        epochLossNotifier.accept(epoch + 1, totalLoss / data.size());
      if (epochMaxDelta <= maxDeltaThreshold)
        break;

      // totalLoss = 0d; // now for validation
      // for (var input : validationData) {
      // var row = input.first();
      // int clsIndex = classesSeen.get(input.second());
      // for (int i = 0; i < rowLength; i++)
      // mainPerceptron.setNetworkInput(i, row.getAsDouble(i));
      // mainPerceptron.processInput();

      // expectedValues[clsIndex] = 1d;
      // for (int i = 0; i < mainPerceptron.outputLayerSize; i++)
      // actualValues[i] = mainPerceptron.getNetworkOutput(i);
      // totalLoss += trainingLossFunction.applyAsDouble(expectedValues,
      // actualValues);
      // expectedValues[clsIndex] = 0d;
      // }

      // if (oldTotalLoss >= 0 && totalLoss > oldTotalLoss)
      // break;
      // oldTotalLoss = totalLoss;
    }
  }

  protected MultiLayerPerceptron perceptron() {
    var pref = threadLocalPerceptrons.get();
    var p = pref == null ? null : pref.get();
    if (p == null) {
      synchronized (this) {
        if (mainPerceptron == null)
          throw new IllegalStateException("finishFitting() not called");
        p = new MultiLayerPerceptron(mainPerceptron);
      }
      threadLocalPerceptrons.set(new SoftReference<>(p));
    }
    return p;
  }

  @Override
  public Optional<C> predict(Float64Row input) {
    final int numClasses = classesSeenList.size();
    if (numClasses == 0)
      return Optional.empty();
    if (numClasses == 1)
      return Optional.of(classesSeenList.get(0));

    final var localPerceptron = perceptron();
    for (int i = 0; i < rowLength; i++)
      localPerceptron.setNetworkInput(i, input.getAsDouble(i));
    localPerceptron.processInput();

    int maxValueIndex = 0;
    double maxValue = localPerceptron.getNetworkOutput(0), current;
    for (int i = 1; i < numClasses; i++) {
      current = localPerceptron.getNetworkOutput(i);
      if (current > maxValue) {
        maxValue = current;
        maxValueIndex = i;
      }
    }

    return Optional.of(classesSeenList.get(maxValueIndex));
  }
}
