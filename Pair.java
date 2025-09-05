import java.util.Map.Entry;

/** A generic pair of values. */
public record Pair<FirstType, SecondType>(FirstType first, SecondType second) implements Entry<FirstType, SecondType> {
  @Override
  public final FirstType getKey() {
    return first;
  }

  @Override
  public final SecondType getValue() {
    return second;
  }

  @Override
  public final SecondType setValue(SecondType value) {
    throw new UnsupportedOperationException();
  }
}