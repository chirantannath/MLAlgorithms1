import java.lang.ref.SoftReference;
import java.util.*;
import java.util.function.*;

public class PerceptronClassifier<C> implements Classifier<Float64Row, C> {

  protected Perceptron mainPerceptron = null;

  public final int rowLength;

  public final double learningRate;

  public final long maxEpochs;

  public final double maxDeltaThreshold;

  public final ActivationFunction activationFunction;

  public final LossFunction lossFunction;

  public final DoubleSupplier initialWeightsGenerator;

  protected List<Pair<Float64Row, C>> data = new ArrayList<>();

  protected Map<C, Integer> classesSeen = new LinkedHashMap<>();

  protected List<C> classesSeenList = new ArrayList<>();

  protected final ThreadLocal<SoftReference<Perceptron>> threadLocalPerceptrons = new ThreadLocal<>();

  protected LongConsumer epochNotifier = null;

  public PerceptronClassifier(
      int rowLength,
      double learningRate,
      long maxEpochs,
      double maxDeltaThreshold,
      ActivationFunction activationFunction,
      LossFunction lossFunction,
      DoubleSupplier initialWeightsGenerator) {
    this.activationFunction = Objects.requireNonNull(activationFunction, "activationFunction");
    this.lossFunction = Objects.requireNonNull(lossFunction, "lossFunction");
    this.initialWeightsGenerator = initialWeightsGenerator == null ? new Random()::nextGaussian
        : initialWeightsGenerator;
    if (rowLength <= 0)
      throw new IllegalArgumentException("rowLength");
    this.rowLength = rowLength;
    if (learningRate <= 0)
      throw new IllegalArgumentException("learningRate");
    this.learningRate = learningRate;
    this.maxEpochs = maxEpochs;
    this.maxDeltaThreshold = maxDeltaThreshold;
  }

  public PerceptronClassifier(
      int rowLength,
      double learningRate,
      long maxEpochs,
      double maxDeltaThreshold,
      ActivationFunction activationFunction,
      LossFunction lossFunction,
      Random initialWeightsGenerator) {
    this(rowLength, learningRate, maxEpochs, maxDeltaThreshold, activationFunction, lossFunction,
        initialWeightsGenerator::nextGaussian);
  }

  public PerceptronClassifier(
      int rowLength,
      double learningRate,
      long maxEpochs,
      double maxDeltaThreshold,
      ActivationFunction activationFunction,
      LossFunction lossFunction) {
    this(rowLength, learningRate, maxEpochs, maxDeltaThreshold, activationFunction, lossFunction,
        new Random()::nextGaussian);
  }

  public PerceptronClassifier(int rowLength, double learningRate, long maxEpochs, double maxDeltaThreshold,
      ActivationFunction activationFunction) {
    this(rowLength, learningRate, maxEpochs, maxDeltaThreshold, activationFunction, LossFunction.SOFTMAX_LOG_LOSS);
  }

  public PerceptronClassifier(int rowLength, double learningRate, long maxEpochs, double maxDeltaThreshold) {
    this(rowLength, learningRate, maxEpochs, maxDeltaThreshold, ActivationFunction.SIGMOID, LossFunction.LOG_LOSS);
  }

  public PerceptronClassifier(int rowLength, double learningRate, long maxEpochs) {
    this(rowLength, learningRate, maxEpochs, 0d);
  }

  public PerceptronClassifier(int rowLength, double learningRate) {
    this(rowLength, learningRate, Short.MAX_VALUE);
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

  public synchronized void setEpochNotifier(LongConsumer notifier) {
    epochNotifier = notifier;
  }

  @Override
  public synchronized void finishFitting() {
    if (data instanceof ArrayList<?> al)
      al.trimToSize();
    data = Collections.unmodifiableList(data);
    classesSeen = Collections.unmodifiableMap(classesSeen);
    if (classesSeenList instanceof ArrayList<?> al)
      al.trimToSize();
    classesSeenList = Collections.unmodifiableList(classesSeenList);
    final int numClasses = classesSeenList.size();

    mainPerceptron = new Perceptron(
        rowLength, // input size
        numClasses, // one output value for each class, in the given order
        activationFunction,
        initialWeightsGenerator);

    double[] expectedValues = new double[mainPerceptron.outputLayerSize];
    Arrays.fill(expectedValues, 0d);
    for (long epoch = 0; epoch < maxEpochs; epoch++) {
      double epochMaxDelta = 0d; // Don't remove this default

      for (var input : data) {
        var row = input.first();
        int clsIndex = classesSeen.get(input.second());
        for (int i = 0; i < rowLength; i++)
          mainPerceptron.setInput(i, row.getAsDouble(i));
        mainPerceptron.processInput();
        expectedValues[clsIndex] = 1d;
        double maxDelta = mainPerceptron.adjustWeights(expectedValues, lossFunction,
            learningRate);
        expectedValues[clsIndex] = 0d;
        epochMaxDelta = Math.max(epochMaxDelta, maxDelta);
      }

      if (epochNotifier != null)
        epochNotifier.accept(epoch + 1);
      if (epochMaxDelta <= maxDeltaThreshold)
        break;
    }
  }

  protected Perceptron perceptron() {
    var pref = threadLocalPerceptrons.get();
    var p = pref == null ? null : pref.get();
    if (p == null) {
      synchronized (this) {
        if (mainPerceptron == null)
          throw new IllegalStateException("finishFitting() not called");
        p = new Perceptron(mainPerceptron);
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

    final Perceptron localPerceptron = perceptron();
    for (int i = 0; i < rowLength; i++)
      localPerceptron.setInput(i, input.getAsDouble(i));
    localPerceptron.processInput();

    int maxValueIndex = 0;
    double maxValue = localPerceptron.getOutput(0), current;
    for (int i = 1; i < numClasses; i++) {
      current = localPerceptron.getOutput(i);
      if (current > maxValue) {
        maxValue = current;
        maxValueIndex = i;
      }
    }

    return Optional.of(classesSeenList.get(maxValueIndex));
  }

}
