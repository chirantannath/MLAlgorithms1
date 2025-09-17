import java.util.*;
import java.util.function.*;
import java.util.stream.*;

/**
 * Default implementation of {@link Row} featuring only {@code double}-type
 * attributes.
 * After creation we can change attributes but not their number.
 * 
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
    data = new double[dataRow.getRowLength()];
    for (int i = 0; i < data.length; i++) {
      try {
        data[i] = dataRow.getAsNumber(i).orElseThrow().doubleValue();
      } catch (NoSuchElementException e) {
        throw new IllegalArgumentException("Not a number column at position " + i + " for row " + dataRow, e);
      }
    }
  }

  @Override
  public String toString() {
    return Arrays.toString(data);
  }

  @Override
  public boolean equals(Object other) {
    if (this == other)
      return true;
    if (other == null)
      return false;
    return other instanceof Float64Row row && Arrays.equals(data, row.data);
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(data);
  }

  public double[] toArray() {
    return Arrays.copyOf(data, data.length);
  }

  public DoubleStream stream() {
    return Arrays.stream(data);
  }

  public double getAsDouble(int index) throws IndexOutOfBoundsException {
    return data[index];
  }

  public void setAsDouble(int index, double value) throws IndexOutOfBoundsException {
    data[index] = value;
  }

  public synchronized void setSynchronized(int index, double value) throws IndexOutOfBoundsException {
    setAsDouble(index, value);
  }

  @Override
  public Class<?>[] getColumnClasses() {
    Class<?>[] classes = new Class<?>[getRowLength()];
    Arrays.fill(classes, Double.TYPE);
    return classes;
  }

  @Override
  public Class<Double> getColumnClass(int index) {
    return Double.TYPE;
  }

  @Override
  public int getRowLength() {
    return data.length;
  }

  @Override
  public Double get(int index) throws IndexOutOfBoundsException {
    return getAsDouble(index);
  }

  @Override
  public Optional<Double> getAsNumber(int index) throws IndexOutOfBoundsException {
    return Optional.of(getAsDouble(index));
  }

  @Override
  public double applyAsDouble(int index) {
    return getAsDouble(index);
  }

  @Override
  public Float64Row project(final int... indices) throws IndexOutOfBoundsException {
    double[] newRow = new double[indices.length];
    for (int i = 0; i < indices.length; i++)
      newRow[i] = getAsDouble(indices[i]);
    return new Float64Row(newRow);
  }

  public Float64Row doBinaryOperation(DoubleBinaryOperator op, Float64Row rightOperand) {
    if (getRowLength() != rightOperand.getRowLength())
      throw new IllegalArgumentException();
    double[] newRow = new double[getRowLength()];
    for (int i = 0; i < newRow.length; i++)
      newRow[i] = op.applyAsDouble(getAsDouble(i), rightOperand.getAsDouble(i));
    return new Float64Row(newRow);
  }

  public Float64Row doUnaryOperation(DoubleUnaryOperator op) {
    double[] newRow = new double[getRowLength()];
    for (int i = 0; i < newRow.length; i++)
      newRow[i] = op.applyAsDouble(getAsDouble(i));
    return new Float64Row(newRow);
  }

  public double distanceEuclidean(Float64Row rightOperand) {
    if (getRowLength() != rightOperand.getRowLength())
      throw new IllegalArgumentException();
    double sum = 0, step;
    final int length = getRowLength();
    for (int i = 0; i < length; i++) {
      step = getAsDouble(i) - rightOperand.getAsDouble(i);
      sum += (step * step);
    }
    return Math.sqrt(sum);
  }

  public double distanceManhattan(Float64Row rightOperand) {
    if (getRowLength() != rightOperand.getRowLength())
      throw new IllegalArgumentException();
    double sum = 0, step;
    final int length = getRowLength();
    for (int i = 0; i < length; i++) {
      step = Math.abs(getAsDouble(i) - rightOperand.getAsDouble(i));
      sum += step;
    }
    return sum;
  }

  public double distanceChebyshev(Float64Row rightOperand) {
    final int length = getRowLength();
    if (length == 0)
      throw new IllegalStateException();
    if (length != rightOperand.getRowLength())
      throw new IllegalArgumentException();
    double max, step;
    max = Math.abs(getAsDouble(0) - rightOperand.getAsDouble(0));
    for (int i = 1; i < length; i++) {
      step = Math.abs(getAsDouble(i) - rightOperand.getAsDouble(i));
      if (step > max)
        max = step;
    }
    return max;
  }

  public double distanceMinkowski(Float64Row rightOperand, final double p) {
    if (getRowLength() != rightOperand.getRowLength())
      throw new IllegalArgumentException();
    double sum = 0, step;
    final int length = getRowLength();
    for (int i = 0; i < length; i++) {
      step = Math.abs(getAsDouble(i) - rightOperand.getAsDouble(i));
      sum += Math.pow(step, p);
    }
    return Math.pow(sum, 1D / p);
  }

  public double dotProduct(Float64Row rightOperand) {
    final int length = getRowLength();
    if (length != rightOperand.getRowLength())
      throw new IllegalArgumentException();
    double sum = 0, li, ri;
    for (int i = 0; i < length; i++) {
      li = getAsDouble(i);
      ri = rightOperand.getAsDouble(i);
      sum += li * ri;
    }
    return sum;
  }

  public double vectorMagnitude() {
    final int length = getRowLength();
    double sum = 0, xi;
    for (int i = 0; i < length; i++) {
      xi = getAsDouble(i);
      sum += xi * xi;
    }
    return Math.sqrt(sum);
  }

  public double distanceCosineSimilarity(Float64Row rightOperand) {
    final int length = getRowLength();
    if (length != rightOperand.getRowLength())
      throw new IllegalArgumentException();
    double sum = 0, li, ri, modLeft = 0, modRight = 0;
    for (int i = 0; i < length; i++) {
      li = getAsDouble(i);
      ri = rightOperand.getAsDouble(i);
      modLeft += li * li;
      modRight += ri * ri;
      sum += li * ri;
    }
    sum /= Math.sqrt(modLeft);
    sum /= Math.sqrt(modRight);
    return sum;
  }

  public double distancePearsonCorrelation(Float64Row rightOperand) {
    final int length = getRowLength();
    if (length != rightOperand.getRowLength())
      throw new IllegalArgumentException();

    double leftMean = 0, rightMean = 0;
    for (int i = 0; i < length; i++) {
      leftMean += (getAsDouble(i) / length);
      rightMean += (rightOperand.getAsDouble(i) / length);
    }

    double leftSSR = 0, rightSSR = 0, residualProductSum = 0, leftResidual, rightResidual;
    for (int i = 0; i < length; i++) {
      leftResidual = getAsDouble(i) - leftMean;
      rightResidual = rightOperand.getAsDouble(i) - rightMean;
      residualProductSum += (leftResidual * rightResidual);
      leftSSR += (leftResidual * leftResidual);
      rightSSR += (rightResidual * rightResidual);
    }

    return 1d - residualProductSum / Math.sqrt(leftSSR * rightSSR);
  }

  public static Float64Row concatenate(Float64Row... rows) {
    final double[][] data = new double[rows.length][];
    for (int i = 0; i < rows.length; i++)
      data[i] = rows[i].data;
    final double[] concatenated = Utils.concat(data);
    return new Float64Row(concatenated);
  }
}
