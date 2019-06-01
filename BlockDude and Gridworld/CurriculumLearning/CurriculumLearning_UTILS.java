package CurriculumLearning;

import BlockDude.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

import CurriculumLearning.CurriculumLearning.CurriculumStep;
import GridWorld.GridWorldTransferLearning_UTILS.GridWorldDomainFeatures;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.mdp.core.state.State;

public class CurriculumLearning_UTILS {
		
		/************************* Curriculum Generator ****************************/
		
		public static class CurriculumGenerator{
			
			ArrayList<Object> 				sourceSet;
			Object 							targetTask;
			HashMap<Integer,ArrayList>		curriculaCollectorIdx;//this member contains all the combination of 
			String 							root;
			
			public CurriculumGenerator(ArrayList<Object> sources, Object target, String root){
				
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
							CurriculumLearning currentCurriculum = new CurriculumLearning(tempCurriculumSources, new CurriculumStep(this.targetTask));
							
							currentCurriculum.performCurriculum(rootFolder);
							
						}
					}		
				}
				
			}
			
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
			
			/** This function instantiate curriculaCollectorIdx*/
			public void createAllCurriculaOnFile(int minLth, int maxLth, String fileName){
				
				PrintWriter collector = null;
				try {
					collector = new PrintWriter(fileName);
				} catch (FileNotFoundException e1) {
					e1.printStackTrace();
				}
				
				for(int curr = minLth; curr <= maxLth; ++curr){
					
					ArrayList<Integer> indexes = genIntArrOfLen(this.sourceSet.size()), 
									   empty   = new ArrayList<Integer>();
					
					ArrayList<ArrayList<Integer>> allIdx = CrazyIndexing(indexes, empty, curr);
					
					for(ArrayList<Integer> thisId : allIdx){
						collector.println(thisId);
					}	
				}
				
				collector.close();
			}
			
			public void performCurriculumFromFile(int numEph, String curriculaFile, int curriculumNumber) {
				
				ArrayList<Integer> currOrder = new ArrayList();
				
				Scanner sc;
				try {
					sc = new Scanner(new File(curriculaFile));
					for(int line = 0; line<curriculumNumber; ++line) {
						sc.nextLine();
					}
					
					Boolean endOfLine = false;
					
					while(sc.hasNext() && !endOfLine) {
						String supp = sc.next().toString();
						for(int l = 0; l<supp.length(); ++l) {
							if(supp.charAt(l) == ']')
								endOfLine = true;
							else if(Character.isDigit(supp.charAt(l))) {
								if(Character.isDigit(supp.charAt(l+1))) {
									currOrder.add(Character.getNumericValue(supp.charAt(l)+supp.charAt(l+1)));
									l++;
								}else
									currOrder.add(Character.getNumericValue(supp.charAt(l)));
							}
						}
					}

					sc.close();
				} catch (FileNotFoundException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
				
				for(int e = 0; e < numEph; ++e){
					String rootFolder = this.root + "Curriculum" + currOrder.toString() + "/CurrEpoch_" + e + "/";
					
					ArrayList<CurriculumStep> tempCurriculumSources = new ArrayList<CurriculumStep>();
					for(int s = 0; s < ((ArrayList)currOrder).size() ; ++s){
						tempCurriculumSources.add(new CurriculumStep(this.sourceSet.get((int)((ArrayList)currOrder).get(s))));//TO CHANGE
					}
					CurriculumLearning currentCurriculum = new CurriculumLearning(tempCurriculumSources, new CurriculumStep(this.targetTask));
					
					currentCurriculum.performCurriculum(rootFolder);
					
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
						
		}
		
		/******************************    END    **********************************/
		
		public static int[] getExeParamsForLearning(Object mapIdentifier){
			int[] params = new int[2];
			
			int numTilings = 0;
			int numEpisodes = 0;
			
			if(mapIdentifier instanceof BlockDudeState) {
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
				}else if(mapIdentifier == BlockDudeTransferLearning_UTILS.initialState_blocks_short_2){
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
			}else if(mapIdentifier instanceof GridWorldDomainFeatures) {
				if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("test")) {
					numTilings = 4;
					numEpisodes = 80;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("test1")) {
					numTilings = 4;
					numEpisodes = 80;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("test2")) {
					numTilings = 4;
					numEpisodes = 80;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("test_L")) {
					numTilings = 4;
					numEpisodes = 300;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("test_Lv2")) {
					numTilings = 4;
					numEpisodes = 300;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("treasure")) {
					numTilings = 4;
					numEpisodes = 30;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("fire")) {
					numTilings = 4;
					numEpisodes = 30;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("pit")) {
					numTilings = 4;
					numEpisodes = 30;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("test_M")) {
					numTilings = 4;
					numEpisodes = 150;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("test_M_1")) {
					numTilings = 4;
					numEpisodes = 150;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("test_M_2")) {
					numTilings = 4;
					numEpisodes = 150;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("test_L_2")) {
					numTilings = 4;
					numEpisodes = 300;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("test_Lv2_2")) {
					numTilings = 4;
					numEpisodes = 300;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("0")) {//not working properly for some reason
					numTilings = 4;
					numEpisodes = 500;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("treasure_XL")) {
					numTilings = 4;
					numEpisodes = 50;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("test_manyF")) {
					numTilings = 4;
					numEpisodes = 100;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("test_M_manyP")) {
					numTilings = 4;
					numEpisodes = 300;
				}else if(((GridWorldDomainFeatures)mapIdentifier).tag.equals("test_L_manyF")) {
					numTilings = 4;
					numEpisodes = 400;
				}
				
			}else 
				System.err.println("ERROR: the state type cannot be handled");
		
			params[0] = numTilings;
			params[1] = numEpisodes;		
			
			return params;
		}
}
