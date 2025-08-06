import java.util.*;
import java.util.function.*;
import java.util.stream.*;
/**
 * Default implementation of {@link Row} featuring only {@code double}-type attributes.
 * After creation we can change attributes but not their number.
 * @author chirantannath
 */
public final class Float64Row implements Row, IntToDoubleFunction {
  private final double[] data;

  public Float64Row(double[] data) {
    Objects.requireNonNull(data);
    this.data = Arrays.copyOf(data, data.length);
  }
  public Float64Row(int length) {
    data = new double[length];
    Arrays.fill(data, 0d);
  }
  /** Tries to convert an arbitrary row into this class. */
  public Float64Row(final Row dataRow) throws IllegalArgumentException {
    data = IntStream.range(0, dataRow.getRowLength())
    .mapToDouble(i -> dataRow.getAsNumber(i).orElseThrow(() -> new IllegalArgumentException("Not a number column at position "+i+" for row "+dataRow)).doubleValue())
    .toArray();
  }

  @Override public String toString() {return Arrays.toString(data);}
  @Override public boolean equals(Object other) {
    if(this == other) return true;
    if(other == null) return false;
    return other instanceof Float64Row row && Arrays.equals(data, row.data);
  }
  @Override public int hashCode() {return Arrays.hashCode(data);}

  public double[] toArray() {return Arrays.copyOf(data, data.length);}
  public DoubleStream stream() {return Arrays.stream(data);}
  public double getAsDouble(int index) throws IndexOutOfBoundsException {return data[index];}
  public void setAsDouble(int index, double value) throws IndexOutOfBoundsException {data[index] = value;}
  public synchronized void setSynchronized(int index, double value) throws IndexOutOfBoundsException {setAsDouble(index, value);}

  @Override public Class<?>[] getColumnClasses() {
    return IntStream.range(0, data.length)
    .mapToObj(i -> (Class<?>)double.class).toArray(Class<?>[]::new);
  }
  @Override public Class<Double> getColumnClass(int index) {return Double.TYPE;}
  @Override public int getRowLength() {return data.length;}
  @Override public Double get(int index) throws IndexOutOfBoundsException {return getAsDouble(index);}
  @Override public Optional<Double> getAsNumber(int index) throws IndexOutOfBoundsException {return Optional.of(getAsDouble(index));}
  @Override public double applyAsDouble(int index) {return getAsDouble(index);}
  @Override public Float64Row project(final int ... indices) throws IndexOutOfBoundsException {
    return new Float64Row(Arrays.stream(indices).mapToDouble(this::getAsDouble).toArray());
  }

  public Float64Row doBinaryOperation(DoubleBinaryOperator op, Float64Row rightOperand) {
    if(getRowLength() != rightOperand.getRowLength()) throw new IllegalArgumentException();
    return new Float64Row(IntStream.range(0, getRowLength())
    .mapToDouble(i -> op.applyAsDouble(getAsDouble(i), rightOperand.getAsDouble(i)))
    .toArray());
  }
  // public Float64Row doBinaryOperationParallel(DoubleBinaryOperator operator, Float64Row rightOperand) {
  //   if(getRowLength() != rightOperand.getRowLength()) throw new IllegalArgumentException();
  //   return new Float64Row(IntStream.range(0, getRowLength()).parallel()
  //   .mapToDouble(i -> operator.applyAsDouble(getAsDouble(i), rightOperand.getAsDouble(i)))
  //   .toArray());
  // }
  
  public Float64Row doUnaryOperation(DoubleUnaryOperator op) {
    return new Float64Row(IntStream.range(0, getRowLength())
    .mapToDouble(i -> op.applyAsDouble(getAsDouble(i)))
    .toArray());
  }
  // public Float64Row doUnaryOperationParallel(DoubleUnaryOperator operator) {
  //   return new Float64Row(IntStream.range(0, getRowLength()).parallel()
  //   .mapToDouble(i -> operator.applyAsDouble(getAsDouble(i)))
  //   .toArray());
  // }

  public double distanceEuclidean(Float64Row rightOperand) {
    if(getRowLength() != rightOperand.getRowLength()) throw new IllegalArgumentException();
    return Math.sqrt(IntStream.range(0, getRowLength())
    .mapToDouble(i -> {
      final var step = getAsDouble(i) - rightOperand.getAsDouble(i);
      return step * step;
    }).unordered().sum());
  }
  public double distanceManhattan(Float64Row rightOperand) {
    if(getRowLength() != rightOperand.getRowLength()) throw new IllegalArgumentException();
    return IntStream.range(0, getRowLength())
    .mapToDouble(i -> Math.abs(getAsDouble(i) - rightOperand.getAsDouble(i))).unordered().sum();
  }
  public double distanceChebyshev(Float64Row rightOperand) {
    if(getRowLength() != rightOperand.getRowLength()) throw new IllegalArgumentException();
    return IntStream.range(0, getRowLength())
    .mapToDouble(i -> Math.abs(getAsDouble(i) - rightOperand.getAsDouble(i))).unordered().max()
    .orElseThrow(IllegalStateException::new);
  }
  public double distanceMinkowski(Float64Row rightOperand, final double p) {
    if(getRowLength() != rightOperand.getRowLength()) throw new IllegalArgumentException();
    return Math.pow(IntStream.range(0, getRowLength())
    .mapToDouble(i -> Math.pow(Math.abs(getAsDouble(i) - rightOperand.getAsDouble(i)), p)).unordered().sum()
    , 1D/p);
  }
}
