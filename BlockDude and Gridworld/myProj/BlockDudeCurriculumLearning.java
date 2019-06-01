package myProj;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;

import GridWorld.GridWorldTransferLearning_UTILS;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import myProj.BlockDudeCurriculumLearning_UTILS.CurriculumGenerator;

public class BlockDudeCurriculumLearning {
	
	ArrayList<CurriculumStep> 		sourceTasks;
	CurriculumStep 					targetTask;
		
	public BlockDudeCurriculumLearning(ArrayList<CurriculumStep> sources, CurriculumStep target){
		
		this.sourceTasks = sources;
		this.targetTask = target;
		
	};
	
	public void performCurriculum(String outputFolder){
		
		TLLinearVFA allPreviousVFA = this.fullCascadeCurriculumLearning(outputFolder);
		
		this.targetTask.learn(outputFolder + "outputTarget/", allPreviousVFA);
		//this.targetTask.runVisualizer();
		
	};	
	
	private TLLinearVFA fullCascadeCurriculumLearning(String outputFolder){//all the elements in the source are used, and the original order is kept
		
		TLLinearVFA previousVFA = null;
		int i = 0;
		
		for(CurriculumStep currSource : this.sourceTasks){
			System.out.println("Source " + i);
			
			String currOut = outputFolder + "outputSource" + i + "/";
			
			previousVFA = currSource.learn(currOut, previousVFA);
			
			++i;
		}
		
		return previousVFA;
	}
	
	/************************* Curriculum Step Class ***************************/
	
	public static class CurriculumStep{
		
		BlockDudeState					taskState;
		int 							numTilings;
		int 							numEpisodes;
		
		BlockDudeTransferLearning		taskTransferLearning 	= null;
		String							outputPath   			= null;
		
		//Basic Constructor
		public CurriculumStep(BlockDudeState source){
			this.taskState = source;
			
			int[] supp = BlockDudeCurriculumLearning_UTILS.getExeParamsForLearning(this.taskState);
			
			this.numTilings = supp[0];
			this.numEpisodes = supp[1];
		}
		
		//Creation of the object BlockDudeTransferLearning with which you can actually perform the learning
		private void initTransferLearning(String outputPath){
			
			this.outputPath = outputPath;
			BlockDudeTransferLearning_UTILS.delete(new File(outputPath));
			
			this.taskTransferLearning = new BlockDudeTransferLearning(this.taskState, this.numTilings);
		}
		
		public TLLinearVFA learn(String outputPath, TLLinearVFA sourceVFA){
			
			this.initTransferLearning(outputPath);
			
			return this.taskTransferLearning.PerformLearningAlgorithm(this.outputPath, 2, 1, this.numEpisodes, sourceVFA);
			
		}
		
		public void runVisualizer(){
			if(this.outputPath != null)
				this.taskTransferLearning.visualize(this.outputPath);
			else
				System.out.println("The Output Path has not been initialized yet therefore you can't visualize the Learning Phase");
		}
	}
	
	/******************************    END    **********************************/
	
	//Some Tests for the Curriculum
	public static void main(String[] args){
		
		//********************** SMALL CURRICULA **************************
//		ArrayList<BlockDudeState> sources = new ArrayList<BlockDudeState>();
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_1);//0
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_2_2);//0
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_2_4);//0
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_3_1);//0
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_3_2);//0
//		
//		CurriculumGenerator gen = new CurriculumGenerator(sources, BlockDudeTransferLearning_UTILS.initialState_4, "Curricula/");//
//		gen.performAllCurricula(100, 10, 0, 3);
//		//curriculum.performCurriculum(100);
		
		//********************** HUGE CURRICULA **************************
//		ArrayList<BlockDudeState> sources = new ArrayList<BlockDudeState>();
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_1_min);//0
//		
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_1);//1
//		
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_2_min);//2
//		
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_2);//3
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_2_1);//4
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_2_2);//5
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_2_3);//6
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_2_4);//7
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_2_5);//8
//		
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_3_min);//9
//		
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_3);//10
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_3_1);//11
//		
//		CurriculumGenerator gen = new CurriculumGenerator(sources, BlockDudeTransferLearning_UTILS.initialState_4, "Curricula/");//
//		gen.performAllCurricula(100, 10, 0, 3);
		//******************************************************************
		
		//********************** HUGE CURRICULA 2 **************************
//		ArrayList<BlockDudeState> sources = new ArrayList<BlockDudeState>();
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_1_min);//0
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_1);//1
//		
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_2_min);//2
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_2);//3
//		
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_pit_min);//4
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_pit);//5
//		
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_reverse_min);//6
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_reverse);//7
//		
//		CurriculumGenerator gen = new CurriculumGenerator(sources, BlockDudeTransferLearning_UTILS.initialState_reverse_trick, "Curricula/");//
//		gen.performAllCurricula(100, 10, 0, 4);		
		//******************************************************************
		
		
//		ArrayList<BlockDudeState> sources = new ArrayList<BlockDudeState>();
//		sources.add(BlockDudeTransferLearning_UTILS.initialState_4);//0
//				
//		CurriculumGenerator gen = new CurriculumGenerator(sources, BlockDudeTransferLearning_UTILS.initialState_4, "TargetComparison/");//
//		gen.performAllCurricula(100, 10, 1, 1);
		//curriculum.performCurriculum(100);
		
		//********************* SPECIFIC CURRICULUM ************************
//		ArrayList<CurriculumStep> sources = new ArrayList<CurriculumStep>();
//		sources.add(new CurriculumStep(BlockDudeTransferLearning_UTILS.initialState_1));
//		sources.add(new CurriculumStep(BlockDudeTransferLearning_UTILS.initialState_2));
//		//sources.add(new CurriculumStep(BlockDudeTransferLearning_UTILS.initialState_2_1));
//		//sources.add(new CurriculumStep(BlockDudeTransferLearning_UTILS.initialState_4));
//		
//		CurriculumStep target = new CurriculumStep(BlockDudeTransferLearning_UTILS.initialState_4);
//		
//		BlockDudeCurriculumLearning bdCL = new BlockDudeCurriculumLearning(sources, target);
//		bdCL.performCurriculum("OUTPUT/Comparison/");
		
		return;
		
	}
	
}
