package myProj;

import java.util.ArrayList;
import java.util.HashMap;

import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.mdp.core.state.State;
import myProj.BlockDudeCurriculumLearning.CurriculumStep;

public class BlockDudeCurriculumLearning_UTILS {
	
	/************************* Curriculum Generator ****************************/
	
	public static class CurriculumGenerator{
		
		ArrayList<BlockDudeState> 		sourceSet;
		BlockDudeState 					targetTask;
		HashMap<Integer,ArrayList>		curriculaCollectorIdx;//this member contains all the combination of 
		String 							root;
		
		public CurriculumGenerator(ArrayList<BlockDudeState> sources, BlockDudeState target, String root){
			
			this.sourceSet = sources;
			this.targetTask = target;
			this.root = root;
			
		};
		
		/** This function first creates all the possible curricula inside the boundaries given
		 *  as inputs (minLth and maxLth), then it performs all of them sequentially
		 *  */
		public void performAllCurricula(int numEp, int numEph, int minLth, int maxLth){
			
			if(maxLth > this.sourceSet.size()){
				System.err.println("The maximum length for the curriculum can't be bigger than the number of Source Tasks given to the Curriculum.");
				return;
			}
				
			createAllCurricula(minLth, maxLth);
			
			for(int currLth : this.curriculaCollectorIdx.keySet()){
				
				System.out.println("Curriculum Length = " + currLth);
				
				for(Object currOrder : this.curriculaCollectorIdx.get(currLth)){
					for(int e = 0; e < numEph; ++e){
						String rootFolder = this.root + "Curriculum" + currOrder.toString() + "/CurrEpoch_" + e + "/";
						
						ArrayList<CurriculumStep> tempCurriculumSources = new ArrayList<CurriculumStep>();
						for(int s = 0; s < ((ArrayList)currOrder).size() ; ++s){
							tempCurriculumSources.add(new CurriculumStep(this.sourceSet.get((int)((ArrayList)currOrder).get(s))));//TO CHANGE
						}
						BlockDudeCurriculumLearning currentCurriculum = new BlockDudeCurriculumLearning(tempCurriculumSources, new CurriculumStep(this.targetTask));
						
						currentCurriculum.performCurriculum(rootFolder);
						
//						TLLinearVFA allPreviousVFA = partialCascadeCurriculumLearning((ArrayList) currOrder, rootFolder);
//					
//						learnTarget(allPreviousVFA, numEp, rootFolder);
					}
				}		
			}
			
		}
		
//		private void learnTarget(TLLinearVFA previousVFA, int numEp, String root){
			
//			CurriculumStep target = new CurriculumStep(this.targetTask, 8, numEp);
//			target.learn(root + "outputTarget/", previousVFA);
//			target.runVisualizer();
			
//			CurriculumStep targetNoTransfer = new CurriculumStep(this.targetTask, 8*this.sourceSet.size(), numEp);
//			targetNoTransfer.learn(root + "outputTargetNoTransfer/", null);
//			targetNoTransfer.runVisualizer();
			
//		}
		
		/** This function instantiate curriculaCollectorIdx*/
		private void createAllCurricula(int minLth, int maxLth){
			
			this.curriculaCollectorIdx = new HashMap<Integer,ArrayList>();
			
			for(int curr = minLth; curr <= maxLth; ++curr){
				
				ArrayList<Integer> indexes = genIntArrOfLen(this.sourceSet.size()), 
								   empty   = new ArrayList<Integer>();
				
				ArrayList<ArrayList<Integer>> allIdx = CrazyIndexing(indexes, empty, curr);
				
				this.curriculaCollectorIdx.put(curr, allIdx);
			}
			
		}
		
		/** This function creates all the combinatorial combinations of indexis we want to analyse */
		private ArrayList<ArrayList<Integer>> CrazyIndexing(ArrayList<Integer> stillToSelect, ArrayList<Integer> alreadySelected, int r){
			
			ArrayList<ArrayList<Integer>> completeCombination = new ArrayList<ArrayList<Integer>>();
			
			if(alreadySelected.size() == r){
				completeCombination.add(alreadySelected);
			}else{
				for(int elem : stillToSelect){
					ArrayList<Integer> newSTS = new ArrayList<Integer>(stillToSelect);
					newSTS.remove(new Integer(elem));
					
					ArrayList<Integer> newAS = new ArrayList<Integer>(alreadySelected);
					newAS.add(elem);
					
					completeCombination.addAll(CrazyIndexing(newSTS, newAS, r));
				}
			}
			
			return completeCombination;
			
		}
		
		/** This function creates a continious array of integers from 0 to len */
		private ArrayList<Integer> genIntArrOfLen(int len){
			ArrayList<Integer> toReturn = new ArrayList<Integer>();
			
			for(int i = 0; i < len; ++i)
				toReturn.add(i);
				
			return toReturn;
		}
		
//		private TLLinearVFA partialCascadeCurriculumLearning(ArrayList<Integer> idOrder, String root){
//			
//			TLLinearVFA previousVFA = null;
//			int i = 0;
//			
//			for(int j : idOrder){
//				String currOut = root + "/" + "outputSource" + i + "_" + j + "/";//Curriculum_361/outputSource0_3
//				
//				previousVFA = this.sourceSet.get(j).learn(currOut, previousVFA);
//				
//				++i;
//			}
//			
//			return previousVFA;
//			
//		}
		
		
	}
	
	/******************************    END    **********************************/
	
	public static int[] getExeParamsForLearning(State mapIdentifier){
		int[] params = new int[2];
		
		int numTilings = 0;
		int numEpisodes = 0;
		
		if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_1_min){
			numTilings = 8;
			numEpisodes = 20;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_1){
			numTilings = 8;
			numEpisodes = 30;//80;//early stopping : ; convergence : 80
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_1_L){
			numTilings = 8;
			numEpisodes = 30;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_2_min){
			numTilings = 8;
			numEpisodes = 30;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_2 || 
				 mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_2_1 ||
				 mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_2_2 ||
				 mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_2_3 ||
				 mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_2_4 ||
				 mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_2_5){
			
			numTilings = 8;
			numEpisodes = 40;//200;//early stopping : ; convergence : 
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_2_L ||
				 mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_2_L2){
			numTilings = 8;
			numEpisodes = 50;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_3_min){
			numTilings = 8;
			numEpisodes = 50;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_3 || 
				 mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_3_1 || 
				 mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_3_2){
			
			numTilings = 8;
			numEpisodes = 100;//250;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_3_L ||
				mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_3_L2){
			numTilings = 8;
			numEpisodes = 150;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_4){
			numTilings = 8;
			numEpisodes = 150;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_pit_min){
			numTilings = 8;
			numEpisodes = 80;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_pit){
			numTilings = 8;
			numEpisodes = 100;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_reverse_min){
			numTilings = 8;
			numEpisodes = 50;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_reverse){
			numTilings = 8;
			numEpisodes = 60;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_reverse_trick){
			numTilings = 8;
			numEpisodes = 150;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_blocks_short){
			numTilings = 8;
			numEpisodes = 5000;//with 100 actions available then sometiimes it solves the problem
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_blocks_pit){
			numTilings = 8;
			numEpisodes = 5000;//with 100 actions available then sometiimes it solves the problem
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_step_min ||
				mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_step){
			numTilings = 8;
			numEpisodes = 40;
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_reuse){
			numTilings = 8;
			numEpisodes = 5000;//with 100 actions available then sometiimes it solves the problem
		}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_useless_bc){
			numTilings = 8;
			numEpisodes = 50;
		}else{
			System.err.println("The chosen map is not implemented in this context");
		}
		
		params[0] = numTilings;
		params[1] = numEpisodes;		
		
		return params;
	}

}
