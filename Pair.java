import java.util.Map.Entry;

/** An input-output combination. */
public record Pair<InputType, OutputType>(InputType input, OutputType output) implements Entry<InputType, OutputType> {
  @Override public final InputType getKey() {return input;}
  @Override public final OutputType getValue() {return output;}
  @Override public final OutputType setValue(OutputType value) {
    throw new UnsupportedOperationException();
  }
}