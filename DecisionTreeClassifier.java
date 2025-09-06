import java.util.*;
import java.util.function.*;
import java.util.stream.*;

public class DecisionTreeClassifier<R extends Row, C> implements Classifier<R, C> {
  /** Actual decision tree. */
  protected final DecisionTree<R, C, C> dtree;

  protected DecisionTreeClassifier(AttrKind[] attrKinds, int depthLimit,
      ToDoubleFunction<? super Map<?, ? extends Number>> impurityFunction,
      Function<? super Stream<Pair<R, C>>, ? extends C> summarizer) {
    dtree = new DecisionTree<>(attrKinds, depthLimit, summarizer, impurityFunction);
  }

  public DecisionTreeClassifier(AttrKind[] attrKinds, int depthLimit,
      ToDoubleFunction<? super Map<?, ? extends Number>> impurityFunction) {
    this(attrKinds, depthLimit, impurityFunction, DecisionTreeClassifier::summarizeResult);
  }

  public DecisionTreeClassifier(AttrKind[] attrKinds, int depthLimit) {
    dtree = new DecisionTree<>(attrKinds, depthLimit, DecisionTreeClassifier::summarizeResult);
  }

  public DecisionTreeClassifier(AttrKind[] attrKinds) {
    this(attrKinds, attrKinds.length + 1);
  }

  public final int getNumAttributes() {
    return dtree.getRowLength();
  }

  public final AttrKind getAttrKind(int colIndex) {
    return dtree.getAttrKind(colIndex);
  }

  public int depthLimit() {
    return dtree.depthLimit;
  }

  public ToDoubleFunction<? super Map<?, ? extends Number>> impurityFunction() {
    return dtree.impurityFunction;
  }

  @Override
  public void fit(R input, C trueOutput) {
    dtree.addDataPoint(input, trueOutput);
  }

  @Override
  public void finishFitting() {
    dtree.buildTree();
  }

  @Override
  public C predict(R input) {
    return dtree.predict(input);
  }

  /** Result summary when we reach the end of a branch. */
  private static <RType extends Row, Cls> Cls summarizeResult(Stream<Pair<RType, Cls>> branchData) {
    return Utils.valueCounts(branchData.unordered().parallel().map(Pair::second))
        .entrySet().parallelStream().unordered()
        .max((e1, e2) -> e1.getValue().compareTo(e2.getValue()))
        .orElseThrow(AssertionError::new).getKey();
  }
}
