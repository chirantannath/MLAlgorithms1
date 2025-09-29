import java.util.Objects;
import java.util.function.DoubleUnaryOperator;

/**
 * Consists of a pair of functions: a normal function from the set of real
 * numbers to the set of real numbers and it's derivative packaged together.
 * This is used for activation functions in ANNs.
 * 
 * @author chirantannath
 */
public interface ActivationFunction extends DoubleUnaryOperator {

  /**
   * The derivative of {@link #applyAsDouble(double)} with respect to it's
   * operand.
   */
  double applyDerivative(double x);

  /**
   * Creates an activation function pair with the function and it's derivative.
   * 
   * @param function   the activation function
   * @param derivative the derivative of {@code function} with respect to its
   *                   argument
   * @return {@code function} and {@code derivative} packaged together
   */
  public static ActivationFunction of(DoubleUnaryOperator function, DoubleUnaryOperator derivative) {
    Objects.requireNonNull(function, "function");
    Objects.requireNonNull(derivative, "derivative");
    return new ActivationFunction() {
      @Override
      public final double applyAsDouble(double x) {
        return function.applyAsDouble(x);
      }

      @Override
      public final double applyDerivative(double x) {
        return derivative.applyAsDouble(x);
      }
    };
  }

  /** Half parabola, 0 when input is less than 0. */
  public static final ActivationFunction HALF_PARABOLA = of(x -> x > 0d ? x * x : 0d, x -> x > 0d ? 2 * x : 0d);

  /** Identity activation function and it's derivative (constant {@code 1.0D}). */
  public static final ActivationFunction IDENTITY = of(DoubleUnaryOperator.identity(), x -> 1d);

  /** Sigmoid function and it's derivative. */
  public static final ActivationFunction SIGMOID = of(x -> 1d / (1d + Math.exp(-x)), x -> {
    final double sigmoid = 1d / (1d + Math.exp(-x));
    return sigmoid * (1 - sigmoid);
  });

  /** Softplus function (the anti-derivative of {@link #SIGMOID}). */
  public static final ActivationFunction SOFTPLUS = of(x -> Math.log(1d + Math.exp(x)), SIGMOID::applyAsDouble);

  /**
   * Rectified Linear Unit (ReLU) function, and its derivative (the Heaviside step
   * function).
   */
  public static final ActivationFunction RECTIFIED_LINEAR_UNIT = of(x -> Math.max(0, x),
      x -> Math.max(0, Math.signum(x)));

  /** Swish function (Sigmoid Linear Unit/SiLU) and it's derivative. */
  public static final ActivationFunction SWISH = of(x -> x * SIGMOID.applyAsDouble(x), x -> {
    final double sigmoid = SIGMOID.applyAsDouble(x);
    return x * sigmoid * (1 - sigmoid) + sigmoid;
  });

  /** Mish function; see arXiv:1908.08681v1. */
  public static final ActivationFunction MISH = of(x -> x * Utils.tanh(SOFTPLUS.applyAsDouble(x)), x -> {
    final double softplus = SOFTPLUS.applyAsDouble(x);
    final double exp = Math.exp(softplus);
    final double mexp = Math.exp(-softplus);
    final double cosh = Math.scalb(exp + mexp, -1);
    final double tanh = Math.scalb(exp - mexp, -1) / cosh;
    return tanh + x * SOFTPLUS.applyDerivative(x) / (cosh * cosh);
  });

  /** Hyperbolic tangent function, and it's derivative. */
  public static final ActivationFunction TANH = of(Utils::tanh, x -> {
    final double tanh = Utils.tanh(x);
    return 1d - tanh * tanh;
  });

  /** Arc tangent function (tangent inverse) and it's derivative. */
  public static final ActivationFunction ATAN = of(Math::atan, x -> 1d / (x * x + 1));
}
