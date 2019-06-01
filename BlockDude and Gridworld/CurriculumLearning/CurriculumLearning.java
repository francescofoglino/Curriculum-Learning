package CurriculumLearning;

import TransferLearning.*;
import BlockDude.*;

import java.io.File;
import java.util.ArrayList;

import CurriculumLearning.CurriculumLearning_UTILS.CurriculumGenerator;
import GridWorld.GridWorldTransferLearning;
import GridWorld.GridWorldTransferLearning_UTILS;
import GridWorld.GridWorldTransferLearning_UTILS.GridWorldDomainFeatures;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.state.State;

public class CurriculumLearning {
		
		ArrayList<CurriculumStep> 		sourceTasks;
		CurriculumStep 					targetTask;
			
		public CurriculumLearning(ArrayList<CurriculumStep> sources, CurriculumStep target){
			
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
			
			Object							taskState;
			int 							numTilings;
			int 							numEpisodes;
			
			TransferLearning				taskTransferLearning 	= null;//TODO
			String							outputPath   			= null;
			
			//Basic Constructor
			public CurriculumStep(Object source){
				this.taskState = source;
				
				int[] supp = CurriculumLearning_UTILS.getExeParamsForLearning(this.taskState);
				
				this.numTilings = supp[0];
				this.numEpisodes = supp[1];
			}
			
			//Creation of the TransferLearning object with which you can actually perform the learning
			private void initTransferLearning(String outputPath){
				
				this.outputPath = outputPath;
				BlockDudeTransferLearning_UTILS.delete(new File(outputPath));//TODO
				
				if(this.taskState instanceof BlockDudeState)
					this.taskTransferLearning = new BlockDudeTransferLearning((BlockDudeState)this.taskState, this.numTilings);//TODO
				else if(this.taskState instanceof GridWorldDomainFeatures)
					this.taskTransferLearning = new GridWorldTransferLearning((GridWorldDomainFeatures)this.taskState, this.numTilings);
				else 
					System.err.println("ERROR: Type of variable taskState doesn't match any of the available types!");
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
			
			//********************** HUGE CURRICULA 2000 **************************
//			ArrayList<State> sources = new ArrayList<State>();
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_1_min);//0
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_1);//1
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2_min);//2
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2);//3
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_pit_min);//4
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_pit);//5
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_reverse_min);//6
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_reverse);//7
//			
//			CurriculumGenerator gen = new CurriculumGenerator(sources, BlockDudeTransferLearning_UTILS.initialState_reverse_trick, "Curricula_TESTS/");//
//			gen.performAllCurricula(100, 10, 0, 4);
			//******************************************************************
			
			//********************** HUGE CURRICULA 1486************************
//			ArrayList<BlockDudeState> sources = new ArrayList<BlockDudeState>();
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_1_min);//0
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_1);//1
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2_min);//2
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2);//3
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2_1);//4
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2_2);//5
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2_3);//6
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2_4);//7
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2_5);//8
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_3_min);//9
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_3);//10
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_3_1);//11
//			
//			CurriculumGenerator gen = new CurriculumGenerator(sources, BlockDudeTransferLearning_UTILS.initialState_4, "Curricula/");//
//			gen.performAllCurricula(100, 10, 0, 3);
			//******************************************************************
			
			//*********************************************************************
			
			//************************* GW CURRICULA 2000 *************************
//			ArrayList<Object> sources = new ArrayList<Object>();
//			
//			sources.add(new GridWorldTransferLearning_UTILS().gwdf_treasure);	//0
//			sources.add(new GridWorldTransferLearning_UTILS().gwdf_fire);		//1
//			sources.add(new GridWorldTransferLearning_UTILS().gwdf_pit);		//2
//			//sources.add(new GridWorldTransferLearning_UTILS().gwdf_test);		//3
//			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test1);		//4
//			//sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_M);	//5
//			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_M_2);	//6
//			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_L_2);	//7
//			
//			CurriculumGenerator gen = new CurriculumGenerator(sources, new GridWorldTransferLearning_UTILS().gwdf_test_L, "CURRICULA/");
//		    //gen.performAllCurricula(0, 10, 0, 4);
			
			//
			//
			//               WOW              WOW              WOW
			//
			//
			//********************** HUGE CURRICULA 1************************
//			ArrayList<Object> sources = new ArrayList<Object>();
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_1_L);//0
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2_L2);//1
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_3_min);//2
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_4);//3
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_pit_min);//4
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_reverse_min);//5
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_step_min);//6
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_step);//7
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_useless_bc);//8
//			
//			CurriculumGenerator gen = new CurriculumGenerator(sources, BlockDudeTransferLearning_UTILS.initialState_pit, "CURRICULA/");//
////			gen.performAllCurricula(100, 10, 0, 5);
//
//			gen.createAllCurriculaOnFile(0, 5, "CurriculaDescription");
////			gen.performCurriculumFromFile(10, "CurriculaDescription", 3611);
			
			//********************** HUGE CURRICULA 2************************
//			ArrayList<Object> sources = new ArrayList<Object>();
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_1_min);//0
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_1);//1
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_1_L);//2
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2_min);//3
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2);//4
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2_5);//5
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_2_L);//6
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_3_min);//7
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_3);//8
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_3_1);//9
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_3_2);//10
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_3_L);//11
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_4);//12
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_pit_min);//13
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_pit);//14
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_reverse_min);//15
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_reverse);//16
//			
//			sources.add(BlockDudeTransferLearning_UTILS.initialState_step_min);//17
//			
//			CurriculumGenerator gen = new CurriculumGenerator(sources, BlockDudeTransferLearning_UTILS.initialState_reverse_trick, "CURRICULA/");//
//
////			gen.createAllCurriculaOnFile(0, 3, "CurriculaDescription");
//			gen.performCurriculumFromFile(10, "CurriculaDescription", 3000);
////			gen.performCurriculumFromFile(10, "/nobackup/scff/burlap/CurriculaDescription", Integer.parseInt(System.getenv("SGE_TASK_ID"))-1);
//			
			//********************** HUGE CURRICULA 3************************
//			ArrayList<Object> sources = new ArrayList<Object>();
//			
//			sources.add(new GridWorldTransferLearning_UTILS().gwdf_treasure);	//0
//			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test2);	//1
//			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_manyF);	//2
//			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_M_manyP);	//3
//			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_M_1);	//4
//			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_L_2);	//5
//			sources.add(new GridWorldTransferLearning_UTILS().gwdf_treasure_XL);	//6
//			
//			CurriculumGenerator gen = new CurriculumGenerator(sources, new GridWorldTransferLearning_UTILS().gwdf_test_L_manyF, "CURRICULA/");//
//
////			gen.createAllCurriculaOnFile(0, 3, "CurriculaDescription");
//			gen.performCurriculumFromFile(10, "CurriculaDescription", 3000);
////			gen.performCurriculumFromFile(10, "/nobackup/scff/burlap/CurriculaDescription", Integer.parseInt(System.getenv("SGE_TASK_ID"))-1);
//			
			//********************** other targets **************************
			
			//exp_3
			ArrayList<Object> sources = new ArrayList<Object>();
			
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test);	//0
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test1);	//1
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test2);	//2
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_L);	//3
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_Lv2);	//4
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_treasure);	//5
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_fire);	//6
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_pit);	//7
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_M);	//8
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_M_1);	//9
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_M_2);	//10
			sources.add(new GridWorldTransferLearning_UTILS().gwdf_test_Lv2_2);	//11
			
			CurriculumGenerator gen = new CurriculumGenerator(sources, new GridWorldTransferLearning_UTILS().gwdf_test_M_2, "../../../temp/otherTargets/exp_3/targets/test_M_2/");//

			gen.createAllCurriculaOnFile(0, 4, "../../../temp/otherTargets/exp_3/CurriculaDescription");
//			gen.performCurriculumFromFile(10, "../../../temp/otherTargets/exp_3/CurriculaDescription", 1);
//			gen.performCurriculumFromFile(10, "/nobackup/scff/burlap/CurriculaDescription", Integer.parseInt(System.getenv("SGE_TASK_ID"))-1);
			
			
			System.out.println("DONE!");
			
			return;
			
		}
}
