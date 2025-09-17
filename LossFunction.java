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

  public static final LossFunction MEAN_SQUARED_ERROR = new LossFunction() {
    @Override
    public final double applyAsDouble(double[] expectedValues, double[] actualValues) {
      if(actualValues.length != expectedValues.length)
        throw new IllegalArgumentException();
      double sum = 0, step;
      for (int i = 0; i < actualValues.length; i++) {
        step = actualValues[i] - expectedValues[i];
        sum += step * step;
      }
      return sum == 0d ? 0d : sum / actualValues.length;
    }

    @Override
    public final double applyDerivativeWRTActual(double[] expectedValues, double[] actualValues, int actualVariableIndex) {
      if(actualValues.length != expectedValues.length)
        throw new IllegalArgumentException();
      return 2d * (actualValues[actualVariableIndex] - expectedValues[actualVariableIndex]) / actualValues.length;
    }
  };

  public static final LossFunction MEAN_ABSOLUTE_ERROR = new LossFunction() {
    @Override
    public final double applyAsDouble(double[] expectedValues, double[] actualValues) {
      if(actualValues.length != expectedValues.length)
        throw new IllegalArgumentException();
      double sum = 0, step;
      for (int i = 0; i < actualValues.length; i++) {
        step = actualValues[i] - expectedValues[i];
        sum += Math.abs(step);
      }
      return sum == 0d ? 0d : sum / actualValues.length;
    }

    @Override
    public final double applyDerivativeWRTActual(double[] expectedValues, double[] actualValues, int actualVariableIndex) {
      if(actualValues.length != expectedValues.length)
        throw new IllegalArgumentException();
      return Math.signum(actualValues[actualVariableIndex] - expectedValues[actualVariableIndex]) / actualValues.length;
    }
  };
}
