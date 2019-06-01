package TransferLearning;

public interface TransferLearning {
	
	/** method that specifies the learning process */
	public TLLinearVFA PerformLearningAlgorithm(String outputPath, int algorithmChoice, int nEpochs, int nEpisodes, TLLinearVFA...sourceVFAs);
	
	/** method for running the visualizer */
	public void visualize(String outputPath);
}
