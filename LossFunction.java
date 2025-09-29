import java.util.function.ToDoubleBiFunction;

/**
 * Loss function for ANNs.
 * 
 * @author chirantannath
 */
public interface LossFunction extends ToDoubleBiFunction<double[], double[]> {
  /**
   * Apply loss function for the expected loss from the expected values versus the
   * actual values.
   */
  @Override
  double applyAsDouble(double[] expectedValues, double[] actualValues);

  /**
   * Derivative of the loss function with respect to a specified actual value
   * variable in the actual values vector.
   */
  double applyDerivativeWRTActual(double[] expectedValues, double[] actualValues, int actualVariableIndex);

  /** Retuns a new loss function that has been scaled by the given factor. */
  public static LossFunction scaledLossFunction(LossFunction lf, double factor) {
    return new LossFunction() {
      @Override
      public final double applyAsDouble(double[] expectedValues, double[] actualValues) {
        return factor * lf.applyAsDouble(expectedValues, actualValues);
      }

      @Override
      public final double applyDerivativeWRTActual(double[] expectedValues, double[] actualValues, int actualVariableIndex) {
        return factor * lf.applyDerivativeWRTActual(expectedValues, actualValues, actualVariableIndex);
      }
    };
  }

  public static final LossFunction SUM_SQUARED_RESIDUALS = new LossFunction() {
    @Override
    public final double applyAsDouble(double[] expectedValues, double[] actualValues) {
      final int length = expectedValues.length;
      double sum = 0, step;
      for (int i = 0; i < length; i++) {
        step = actualValues[i] - expectedValues[i];
        sum += step * step;
      }
      return sum;
    }

    @Override
    public final double applyDerivativeWRTActual(double[] expectedValues, double[] actualValues, int actualVariableIndex) {
      return 2d * (actualValues[actualVariableIndex] - expectedValues[actualVariableIndex]);
    }
  };

  public static final LossFunction SUM_ABSOLUTE_RESIDUALS = new LossFunction() {
    @Override
    public final double applyAsDouble(double[] expectedValues, double[] actualValues) {
      final int length = expectedValues.length;
      double sum = 0, step;
      for (int i = 0; i < length; i++) {
        step = actualValues[i] - expectedValues[i];
        sum += Math.abs(step);
      }
      return sum;
    }

    @Override
    public final double applyDerivativeWRTActual(double[] expectedValues, double[] actualValues, int actualVariableIndex) {
      return Math.signum(actualValues[actualVariableIndex] - expectedValues[actualVariableIndex]);
    }
  };

  public static final LossFunction LOG_LOSS = new LossFunction() {
    @Override
    public final double applyAsDouble(double[] expectedValues, double[] actualValues) {
      final int length = expectedValues.length;
      double sum = 0, step;
      for (int i = 0; i < length; i++) {
        //to avoid taking log of 0
        step = expectedValues[i] == 0 ? 0 : -expectedValues[i] * Math.log(actualValues[i]);
        sum += step;
      }
      return sum;
    }

    @Override
    public final double applyDerivativeWRTActual(double[] expectedValues, double[] actualValues, int actualVariableIndex) {
      return -expectedValues[actualVariableIndex] / actualValues[actualVariableIndex];
    }
  };

  public static final LossFunction SOFTMAX_LOG_LOSS = new LossFunction() {
    @Override
    public final double applyAsDouble(double[] expectedValues, double[] actualValues) {
      final int length = expectedValues.length;
      double softmaxSum = 0, sum = 0, evsum = 0;
      for (int i = 0; i < length; i++) {
        evsum += expectedValues[i];
        sum += -expectedValues[i] * actualValues[i];
        softmaxSum += Math.exp(actualValues[i]);
      }
      return sum + evsum * Math.log(softmaxSum);
    }

    @Override
    public final double applyDerivativeWRTActual(double[] expectedValues, double[] actualValues, int actualVariableIndex) {
      final int length = expectedValues.length;
      double expActual = 0, softmaxSum = 0, evsum = 0, step;
      for(int i = 0; i < length; i++) {
        evsum += expectedValues[i];
        step = Math.exp(actualValues[i]);
        softmaxSum += step;
        if (i == actualVariableIndex) expActual = step;
      }
      return -expectedValues[actualVariableIndex] + evsum * expActual / softmaxSum;
    }
    
  };
}
