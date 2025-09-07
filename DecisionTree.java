import java.io.*;
import java.util.*;
import java.util.function.*;
import java.util.stream.*;

/**
 * Decision Tree main building class. Data points are {@link Pair}s of
 * {@code R}s and {@code IntermediateType}s, and the decision tree must arrive
 * at a result of type ResultType.
 * 
 * @author chirantannath
 */
@SuppressWarnings("unused")
public class DecisionTree<R extends Row, IntermediateType, ResultType> {
  protected abstract sealed class Node permits AttrNode, ResultNode {
    /** Parent from which this node came from. */
    public final Node parent;
    /**
     * Filtering condition on which this branch was taken; constant predicate for
     * root nodes.
     */
    public final Predicate<R> branchFilter;
    /**
     * A description for the branch taken by {@link #branchFilter}, must never be
     * null.
     */
    public final String branchDescription;
    /** Depth of this node in tree, 0 for root. */
    public final int depth;
    /** Filled in during training. */
    protected Collection<? extends Node> children = null;

    public Node(Node parent, String branchDescription, Predicate<R> branchFilter) {
      this.parent = parent;
      this.branchFilter = branchFilter == null ? Utils.constantPredicate(true) : branchFilter;
      this.branchDescription = branchDescription == null ? "" : branchDescription;
      depth = parent == null ? 0 : parent.depth + 1;
    }

    public Node(Node parent, Predicate<R> branchFilter) {
      this(parent, null, branchFilter);
    }

    public Node(Node parent) {
      this(parent, null, null);
    }

    public Node() {
      this(null);
    }

    public final boolean isRoot() {
      return parent == null;
    }

    public abstract boolean isChild();

    /** This function is for use during training. */
    public final boolean filterFromRoot(R row) {
      for (Node node = this; node != null; node = node.parent)
        if (!node.branchFilter.test(row))
          return false;
      return true;
    }

    /**
     * This function is for use during building the tree. Return all possible child
     * branches (processing may happen here, and the result may be cached by
     * implementations, this is not subject to change). Must always return some
     * collection (even if it is of length 0). Elements returned are pairs where the
     * first element is a descriptive string (which may be empty "" but not null).
     */
    public abstract Collection<Pair<String, Predicate<R>>> getAllChildBranches();

    protected void walkTree(Appendable out) throws IOException {
      final String prefix;
      {
        final var pb = new StringBuilder();
        for (int i = 0; i < depth; i++)
          pb.append('\t');
        prefix = pb.toString();
      }
      out.append(String.format("%s-(%s)-%s%s", prefix, branchDescription, toString(), System.lineSeparator()));
      final var currentChildren = children;
      if (currentChildren != null)
        for (var child : currentChildren)
          child.walkTree(out);
    }
  }

  /** Nodes that split on the value of a single attribute. */
  protected abstract sealed class AttrNode extends Node permits CategoricalAttrNode, RealAttrNode {
    public final int attrIndex;

    public AttrNode(int attrIndex, Node parent, String branchDescription, Predicate<R> branchFilter) {
      super(parent, branchDescription, branchFilter);
      Objects.checkIndex(attrIndex, getRowLength());
      this.attrIndex = attrIndex;
    }

    public AttrNode(int attrIndex, Node parent, Predicate<R> branchFilter) {
      this(attrIndex, parent, null, branchFilter);
    }

    public AttrNode(int attrIndex, Node parent) {
      this(attrIndex, parent, null);
    }

    public AttrNode(int attrIndex) {
      this(attrIndex, null);
    }

    @Override
    public String toString() {
      return getColumnName(attrIndex);
    }
  }

  /** Nodes splitting on categorical (discrete) value attributes. */
  protected final class CategoricalAttrNode extends AttrNode {
    protected final Map<?, Predicate<R>> categories;

    public CategoricalAttrNode(int attrIndex, Node parent, String branchDescription, Predicate<R> branchFilter,
        Set<?> attrValues) {
      super(attrIndex, parent, branchDescription, branchFilter);
      Objects.requireNonNull(attrValues, "attrValues");
      final Map<Object, Predicate<R>> branches = new HashMap<>();
      for (final var value : attrValues)
        branches.put(value, row -> Objects.equals(row.get(attrIndex), value));
      categories = Collections.unmodifiableMap(branches);
    }

    public CategoricalAttrNode(int attrIndex, Node parent, String branchDescription, Predicate<R> branchFilter) {
      this(attrIndex, parent, branchDescription, branchFilter,
          categoricalValueCounts(attrIndex).keySet());
    }

    public CategoricalAttrNode(int attrIndex, Node parent, Predicate<R> branchFilter) {
      this(attrIndex, parent, null, branchFilter);
    }

    public CategoricalAttrNode(int attrIndex, Node parent) {
      this(attrIndex, parent, null);
    }

    public CategoricalAttrNode(int attrIndex) {
      this(attrIndex, null);
    }

    @Override
    public boolean isChild() {
      return false;
    }

    @Override
    public Collection<Pair<String, Predicate<R>>> getAllChildBranches() {
      return categories.entrySet().stream().map(e -> new Pair<>(e.getKey().toString(), e.getValue()))
          .collect(Collectors.toUnmodifiableList());
    }
  }

  /**
   * Nodes splitting on continuous (real) value attributes. UNIMPLEMENTED AS OF
   * NOW.
   */
  protected final class RealAttrNode extends AttrNode {
    protected Float64Interval[] bins;

    public RealAttrNode(int attrIndex, Node parent, String branchDescription, Predicate<R> branchFilter, double min,
        double max) {
      this(attrIndex, parent, branchDescription, branchFilter, new Float64Interval(min, max));
    }

    public RealAttrNode(int attrIndex, Node parent, String branchDescription, Predicate<R> branchFilter,
        DoubleSummaryStatistics stats) {
      this(attrIndex, parent, branchDescription, branchFilter, stats.getMin(), stats.getMax());
    }

    public RealAttrNode(int attrIndex, Node parent, String branchDescription, Predicate<R> branchFilter,
        Float64Interval interval) {
      super(attrIndex, parent, branchDescription, branchFilter);
      bins = interval.split(realAttributeSplits);
    }

    public RealAttrNode(int attrIndex, Node parent, String branchDescription, Predicate<R> branchFilter) {
      this(attrIndex, parent, branchDescription, branchFilter,
          rootData.parallelStream().unordered()
              .mapToDouble(p -> p.first().getAsNumber(attrIndex).orElseThrow().doubleValue()).summaryStatistics());
    }

    public RealAttrNode(int attrIndex, Node parent) {
      this(attrIndex, parent, null, null);
    }

    public RealAttrNode(int attrIndex) {
      this(attrIndex, null);
    }

    @Override
    public boolean isChild() {
      return false;
    }

    @Override
    public Collection<Pair<String, Predicate<R>>> getAllChildBranches() {
      return Arrays.stream(bins)
          .map(b -> new Pair<String, Predicate<R>>(String.format("[%f,%f]", b.lowerBound(), b.higherBound()),
              r -> b.testForOptional(r.getAsNumber(attrIndex))))
          .collect(Collectors.toUnmodifiableList());
    }
  }

  protected final class ResultNode extends Node {
    /** The result stored. May be {@code null} or any other arbitrary object. */
    public final ResultType result;

    public ResultNode(ResultType result, Node parent, String branchDescription, Predicate<R> branchFilter) {
      super(parent, branchDescription, branchFilter);
      this.result = result;
    }

    public ResultNode(ResultType result, Node parent) {
      this(result, parent, null, null);
    }

    public ResultNode(ResultType result) {
      this(result, null);
    }

    public ResultNode() {
      this(null, null, null, null);
    }

    /** Creates an empty "stop" node. */
    public ResultNode(Node parent) {
      this(null, parent, null, null);
    }

    @Override
    public boolean isChild() {
      return true;
    }

    @Override
    public Collection<Pair<String, Predicate<R>>> getAllChildBranches() {
      return Collections.emptySet();
    }

    @Override
    public String toString() {
      return Objects.toString(result);
    }
  }

  /** All the known data for which to build tree. */
  protected final List<Pair<R, IntermediateType>> rootData = new ArrayList<>();
  /** Attribute types of the rows. */
  protected final AttrKind[] attrKinds;
  /** Column names, may simply be index numbers if not specified. */
  protected String[] columnNames;
  /** Depth limit to which the tree must be built. */
  public final int depthLimit;
  /** Number of splits/bins to be done for real value attributes. */
  public final int realAttributeSplits;
  /** Minimum number of samples to use for splitting a node (except the root). */
  public final int minSamplesToSplit;
  /**
   * Impurity function that measures the <i>lack</i> of information from a set of
   * (arbitrary) keys to weights.
   */
  public final ToDoubleFunction<? super Map<?, ? extends Number>> impurityFunction;
  /**
   * Summarizer function that will reduce/summarize a subset of known data points
   * into a single result.
   */
  public final Function<? super Stream<Pair<R, IntermediateType>>, ? extends ResultType> summarizer;

  /** The decision tree root. */
  protected Node treeRoot = null; // Will be assigned later

  public DecisionTree(AttrKind[] attrKinds, String[] columnNames, int depthLimit, int realAttributeSplits,
      int minSamplesToSplit,
      Function<? super Stream<Pair<R, IntermediateType>>, ? extends ResultType> summarizer,
      ToDoubleFunction<? super Map<?, ? extends Number>> impurityFunction) {
    Objects.requireNonNull(attrKinds, "attrKinds");
    Objects.requireNonNull(impurityFunction, "impurityFunction");
    Objects.requireNonNull(summarizer, "summarizer");
    this.attrKinds = Arrays.copyOf(attrKinds, attrKinds.length);
    if (columnNames == null)
      columnNames = new String[0];
    this.columnNames = new String[attrKinds.length];
    for (int i = 0; i < attrKinds.length; i++)
      this.columnNames[i] = i < columnNames.length ? columnNames[i] : String.valueOf(i);
    if (depthLimit <= 0)
      throw new IllegalArgumentException("depthLimit");
    if (realAttributeSplits < 3)
      throw new IllegalArgumentException("realAttributeSplits");
    this.depthLimit = depthLimit;
    this.realAttributeSplits = realAttributeSplits;
    this.minSamplesToSplit = Math.max(1, minSamplesToSplit);
    this.impurityFunction = impurityFunction;
    this.summarizer = summarizer;
  }

  public DecisionTree(AttrKind[] attrKinds, int depthLimit, int realAttributeSplits,
      Function<? super Stream<Pair<R, IntermediateType>>, ? extends ResultType> summarizer,
      ToDoubleFunction<? super Map<?, ? extends Number>> impurityFunction) {
    this(attrKinds, null, depthLimit, realAttributeSplits, 1, summarizer, impurityFunction);
  }

  public DecisionTree(AttrKind[] attrKinds, int depthLimit, int realAttributeSplits,
      Function<? super Stream<Pair<R, IntermediateType>>, ? extends ResultType> summarizer) {
    this(attrKinds, depthLimit, realAttributeSplits, summarizer,
        m -> Utils.countedEntropy(m.values().parallelStream().unordered()));
  }

  public void addDataPoint(R input, IntermediateType intermediate) {
    rootData.add(new Pair<>(input, intermediate));
  }

  /** Get a part of the root dataset. */
  protected final Stream<Pair<R, IntermediateType>> filteredData(final Predicate<R> filter) {
    return rootData
        // .stream()
        .parallelStream() // Currently disabled for debugging
        .filter(p -> filter.test(p.first()));
  }

  /** FOR CATEGORICAL VALUES ONLY, get all value counts from full root dataset. */
  protected final Map<?, Long> categoricalValueCounts(int attrIndex) {
    Map<?, Long> result;
    if ((result = categoricalValueCountsCache.get(attrIndex)) != null)
      return result;
    result = Utils.valueCounts(rootData.parallelStream().unordered().map(p -> p.first().get(attrIndex)));
    categoricalValueCountsCache.put(attrIndex, result);
    return result;
  }

  protected final Map<Integer, Map<?, Long>> categoricalValueCountsCache = new HashMap<>();

  /** On which attribute should I split this set? */
  protected int findSplittingAttribute(final Predicate<R> setMembership, Iterator<Integer> attributeIndexes) {
    final var totalLength = filteredData(setMembership).unordered().count();
    if (totalLength == 0)
      return -1;
    final double rootImpurity = impurityFunction
        .applyAsDouble(
            Utils.valueCounts(filteredData(setMembership).unordered().map(Pair::second)));

    if (attributeIndexes == null)
      attributeIndexes = IntStream.range(0, getRowLength()).boxed().iterator();

    double currentGain = Double.NEGATIVE_INFINITY;
    int maxGainIndex = -1;
    while (attributeIndexes.hasNext()) {
      final int attrIndex = attributeIndexes.next();
      double attrImpurity = 0;
      switch (attrKinds[attrIndex]) {
        case CONTINUOUS -> {
          final var stats = filteredData(setMembership).unordered()
              .mapToDouble(p -> p.first().getAsNumber(attrIndex)
                  .orElseThrow(() -> new IllegalArgumentException("Expected number")).doubleValue())
              .summaryStatistics();
          final var bins = new Float64Interval(stats.getMin(), stats.getMax()).split(realAttributeSplits);
          final var attrValueCounts = Utils.valueCounts(
              filteredData(setMembership).map(
                  p -> Arrays.stream(bins)
                      .filter(b -> b.testForOptional(p.first().getAsNumber(attrIndex)))
                      .findAny().orElseThrow(AssertionError::new)));
          for (final var avcount : attrValueCounts.entrySet())
            attrImpurity += impurityFunction
                .applyAsDouble(Utils.valueCounts(
                    filteredData(setMembership.and(r -> avcount.getKey().testForOptional(r.getAsNumber(attrIndex))))
                        .unordered().map(Pair::second)))
                * avcount.getValue() / totalLength;
        }
        case CATEGORICAL -> {
          // Split on attribute
          final var attrValueCounts = Utils
              .valueCounts(filteredData(setMembership).map(p -> p.first().get(attrIndex)).unordered());
          for (final var avcount : attrValueCounts.entrySet())
            attrImpurity += impurityFunction
                .applyAsDouble(Utils.valueCounts(
                    filteredData(setMembership.and(r -> Objects.equals(r.get(attrIndex), avcount.getKey())))
                        .unordered().map(Pair::second)))
                * avcount.getValue() / totalLength;
          // break; Not required in rule switch
        }
      }
      final double gain = rootImpurity - attrImpurity; // impurity should decrease
      if (gain > currentGain) {
        currentGain = gain;
        maxGainIndex = attrIndex;
      }
    }

    return maxGainIndex;
  }

  /**
   * Build tree given root, current level number and attribute set already
   * selected. Note that {@code attributesSelected} needs to be a modifiable set.
   */
  protected void buildTree(final Node root, final Set<Integer> attributesSelected) {
    if (root == null || root.isChild())
      return;
    final Collection<Integer> attributesToBranch = IntStream.range(0, getRowLength()).boxed()
        .filter(i -> !attributesSelected.contains(i))
        .collect(Collectors.toUnmodifiableSet());
    final var children = new ArrayList<Node>();
    final int childDepth = root.depth + 1;
    for (final var branchPair : root.getAllChildBranches()) {
      final String branchDescription = branchPair.first();
      final var branch = branchPair.second();
      final var branchDataFilter = branch.and(root::filterFromRoot);
      // Check if we have at least minSamplesToSplit data points in this branch
      if (filteredData(branchDataFilter).unordered().findAny().isEmpty())
        continue;
      final Node node;
      if (childDepth < depthLimit && !attributesToBranch.isEmpty()
          && filteredData(branchDataFilter).unordered().skip(minSamplesToSplit - 1).findAny().isPresent()
      // this last condition checks whether we have enough samples to split here.
      ) { // I can split further from here.
        final int attrIndex = findSplittingAttribute(branchDataFilter, attributesToBranch.iterator());
        if (attrIndex < 0)
          continue;
        node = switch (attrKinds[attrIndex]) {
          case CONTINUOUS ->
            new RealAttrNode(attrIndex, root, branchDescription, branch, filteredData(branchDataFilter).unordered()
                .mapToDouble(p -> p.first().getAsNumber(attrIndex).orElseThrow().doubleValue()).summaryStatistics());
          case CATEGORICAL -> new CategoricalAttrNode(attrIndex, root, branchDescription, branch);
        };
      } else { // Result node calculation, either on reaching depth or when no more attributes
               // to branch on
        final var result = summarizer.apply(filteredData(branchDataFilter));
        node = new ResultNode(result, root, branchDescription, branch);
      }
      children.add(node);
    }
    assert !children.isEmpty();
    root.children = Collections.unmodifiableCollection(children);

    // Recursively build tree further.
    for (final Node child : children) {
      if (child instanceof AttrNode attrNode) {
        attributesSelected.add(attrNode.attrIndex);
        buildTree(attrNode, attributesSelected);
        attributesSelected.remove(attrNode.attrIndex);
      }
    }
  }

  public void buildTree() {
    // Step 1. Find a splitting attribute.
    final int rootAttrIndex = findSplittingAttribute(Utils.constantPredicate(true), null);
    assert rootAttrIndex >= 0;
    treeRoot = switch (attrKinds[rootAttrIndex]) {
      case CONTINUOUS -> new RealAttrNode(rootAttrIndex);
      case CATEGORICAL -> new CategoricalAttrNode(rootAttrIndex);
    };
    final var attributesSelected = new HashSet<Integer>();
    attributesSelected.add(rootAttrIndex);
    buildTree(treeRoot, attributesSelected);
  }

  public ResultType decide(final R input) throws NoSuchElementException {
    if (treeRoot == null)
      throw new IllegalStateException();
    Node current = treeRoot;
    while (true) {
      if (current instanceof ResultNode r)
        return r.result;
      assert current != null && !current.isChild();
      Node newCurrent = current.children
          .stream() // No need to parallelize here.
          // .parallelStream()
          .unordered()
          .filter(child -> child.branchFilter.test(input))
          .findAny().orElseThrow(() -> new NoSuchElementException("Unable to predict for " + input));
      assert !Objects.equals(current, newCurrent);
      current = newCurrent;
    }
  }

  public int getRowLength() {
    return attrKinds.length;
  }

  public AttrKind getAttrKind(int colIndex) {
    return attrKinds[colIndex];
  }

  public String getColumnName(int colIndex) {
    return columnNames[colIndex];
  }

  public void walkTree(Appendable out) throws IOException {
    if (treeRoot != null)
      treeRoot.walkTree(out);
  }
}
