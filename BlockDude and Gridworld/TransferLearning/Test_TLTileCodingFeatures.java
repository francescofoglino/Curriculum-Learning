package TransferLearning;

import BlockDude.*;

import java.util.ArrayList;

import org.junit.Test;

import GridWorld.GridWorldTransferLearning_UTILS;
import burlap.mdp.core.state.State;
import junit.framework.TestCase;

public class Test_TLTileCodingFeatures extends TestCase {
	
	/************************************ BLOCK DUDE ***************************************/
	
	/** source_T = target_T */
	@Test
	public void test_BD_0() {
		//init source
		State sIS_0 = BlockDudeTransferLearning_UTILS.initialState_1;
		int sNT_0 = 1;
		
		//init target
		State tIS_0 = BlockDudeTransferLearning_UTILS.initialState_1;
		int tNT_0 = 1;
		
		ArrayList<Integer> results_0 = testSupporter_S1(sIS_0, sNT_0, tIS_0, tNT_0);
		
		System.out.println("Not Covered = " + results_0.get(0));
		assertEquals((int)results_0.get(0), tNT_0);
		System.out.println("Covered = " + results_0.get(1));
		assertEquals((int)results_0.get(1), sNT_0);
	}
	
	/** domain_source < domain_target */
	@Test
	public void test_BD_1() {
		//init source
		State sIS_1 = BlockDudeTransferLearning_UTILS.initialState_1;
		int sNT_1 = 1;
		
		//init target
		State tIS_1 = BlockDudeTransferLearning_UTILS.initialState_1_L;
		int tNT_1 = 1;
		
		ArrayList<Integer> results_1 = testSupporter_S1(sIS_1, sNT_1, tIS_1, tNT_1);
		
		System.out.println("Not Covered = " + results_1.get(0));
		assertEquals((int)results_1.get(0), tNT_1);
		System.out.println("Covered = " + results_1.get(1));
		assertEquals((int)results_1.get(1), sNT_1);
	}
	
	/** stateVar_source != stateVar_target */
	@Test
	public void test_BD_2() {
		//init source
		State sIS_2 = BlockDudeTransferLearning_UTILS.initialState_1;
		int sNT_2 = 1;
		
		//init target
		State tIS_2 = BlockDudeTransferLearning_UTILS.initialState_3;
		int tNT_2 = 1;
		
		ArrayList<Integer> results_2 = testSupporter_S1(sIS_2, sNT_2, tIS_2, tNT_2);
		
		System.out.println("Not Covered = " + results_2.get(0));
		assertEquals((int)results_2.get(0), tNT_2);
		System.out.println("Covered = " + results_2.get(1));
		assertEquals((int)results_2.get(1), sNT_2);
	}
	
	/** |stateVar_source| != |stateVar_target| &&  domain_source < domain_target */
	@Test
	public void test_BD_3() {
		//init source
		State sIS_3 = BlockDudeTransferLearning_UTILS.initialState_1;
		int sNT_3 = 1;
		
		//init target
		State tIS_3 = BlockDudeTransferLearning_UTILS.initialState_2_L;
		int tNT_3 = 1;
		
		ArrayList<Integer> results_3 = testSupporter_S1(sIS_3, sNT_3, tIS_3, tNT_3);
		
		System.out.println("Not Covered = " + results_3.get(0));
		assertEquals((int)results_3.get(0), tNT_3);
		System.out.println("Covered = " + results_3.get(1));//gneeeeeeeeeeeeeeee
		assertEquals((int)results_3.get(1), sNT_3);
	}
	
	/** |stateVar_source| != |stateVar_target| &&  domain_source < domain_target */
	@Test
	public void test_BD_3_W() {
		//init source
		State sIS_3 = BlockDudeTransferLearning_UTILS.initialState_3;
		int sNT_3 = 1;
		
		//init target
		State tIS_3 = BlockDudeTransferLearning_UTILS.initialState_4;
		int tNT_3 = 1;
		
		ArrayList<Integer> results_3 = testSupporter_S1(sIS_3, sNT_3, tIS_3, tNT_3);
		
		System.out.println("Not Covered = " + results_3.get(0));
		assertEquals((int)results_3.get(0), tNT_3);
		System.out.println("Covered = " + results_3.get(1));//gneeeeeeeeeeeeeeee
		assertEquals((int)results_3.get(1), sNT_3);
	}
	
	/** two different sources in cascade */
	@Test
	public void test_BD_4() {
		//init source 1
		State sIS_4_1 = BlockDudeTransferLearning_UTILS.initialState_1;
		int sNT_4_1 = 1;
		
		//init source 2
		State sIS_4_2 = BlockDudeTransferLearning_UTILS.initialState_2;
		int sNT_4_2 = 1;
		
		//init target
		State tIS_4 = BlockDudeTransferLearning_UTILS.initialState_2_L;
		int tNT_4 = 1;
		
		ArrayList<Integer> results_4 = testSupporter_S2(sIS_4_1, sNT_4_1, sIS_4_2, sNT_4_2, tIS_4, tNT_4);
		
		System.out.println("Not Covered = " + results_4.get(0));
		assertEquals((int)results_4.get(0), tNT_4);
		System.out.println("Covered = " + results_4.get(1));
		assertEquals((int)results_4.get(1), sNT_4_1 + sNT_4_2);
	}
	
	/************************************ GRID WORLD ***************************************/
	
	/** source_T = target_T */
	@Test
	public void test_GW_0() {
		//init source
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source = new GridWorldTransferLearning_UTILS().gwdf_treasure;
		int sNT_0 = 1;
		
		//init target
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_transfer = new GridWorldTransferLearning_UTILS().gwdf_treasure;
		int tNT_0 = 1;
		
		ArrayList<Integer> results_0 = testSupporter_S1_GW(gwdf_source, sNT_0, gwdf_transfer, tNT_0);
		
		System.out.println("Not Covered = " + results_0.get(0));
		assertEquals((int)results_0.get(0), tNT_0);
		System.out.println("Covered = " + results_0.get(1));
		assertEquals((int)results_0.get(1), sNT_0);
	}
	
	
	/** domain_source != domain_target */
	//domain_source < domain_target
	@Test
	public void test_GW_1_m() {
		//init source
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source = new GridWorldTransferLearning_UTILS().gwdf_treasure;
		int sNT_1 = 1;
		
		//init target
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_transfer = new GridWorldTransferLearning_UTILS().gwdf_treasure_L;
		int tNT_1 = 1;
		
		ArrayList<Integer> results_1 = testSupporter_S1_GW(gwdf_source, sNT_1, gwdf_transfer, tNT_1);
		
		System.out.println("Not Covered = " + results_1.get(0));
		assertEquals((int)results_1.get(0), tNT_1);
		System.out.println("Covered = " + results_1.get(1));
		assertEquals((int)results_1.get(1), sNT_1);
	}
	
	
	//domain_source > domain_target
	@Test
	public void test_GW_1_M() {
		//init source
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source = new GridWorldTransferLearning_UTILS().gwdf_treasure_L;
		int sNT_1 = 1;
		
		//init target
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_transfer = new GridWorldTransferLearning_UTILS().gwdf_treasure;
		int tNT_1 = 1;
		
		ArrayList<Integer> results_1 = testSupporter_S1_GW(gwdf_source, sNT_1, gwdf_transfer, tNT_1);
		
		System.out.println("Not Covered = " + results_1.get(0));
		assertEquals((int)results_1.get(0), tNT_1);
		System.out.println("Covered = " + results_1.get(1));
		assertEquals((int)results_1.get(1), sNT_1);
	}
	
	
	/** |stateVar_source| != |stateVar_target| */
	//|stateVar_source| < |stateVar_target|
	
	//fire version
	@Test
	public void test_GW_2_F_m() {
		//init source
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source = new GridWorldTransferLearning_UTILS().gwdf_treasure;
		int sNT_2 = 1;
		
		//init target
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_transfer = new GridWorldTransferLearning_UTILS().gwdf_fire;
		int tNT_2 = 1;
		
		ArrayList<Integer> results_2 = testSupporter_S1_GW(gwdf_source, sNT_2, gwdf_transfer, tNT_2);
		
		System.out.println("Not Covered = " + results_2.get(0));
		assertEquals((int)results_2.get(0), tNT_2);
		System.out.println("Covered = " + results_2.get(1));
		assertEquals((int)results_2.get(1), sNT_2);
	}
	
	//pit version
	@Test
	public void test_GW_2_P_m() {
		//init source
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source = new GridWorldTransferLearning_UTILS().gwdf_treasure;
		int sNT_2 = 1;
		
		//init target
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_transfer = new GridWorldTransferLearning_UTILS().gwdf_pit;
		int tNT_2 = 1;
		
		ArrayList<Integer> results_2 = testSupporter_S1_GW(gwdf_source, sNT_2, gwdf_transfer, tNT_2);
		
		System.out.println("Not Covered = " + results_2.get(0));
		assertEquals((int)results_2.get(0), tNT_2);
		System.out.println("Covered = " + results_2.get(1));
		assertEquals((int)results_2.get(1), sNT_2);
	}
	
	//fire and pit version
	@Test
	public void test_GW_2_FP_m() {
		//init source
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source = new GridWorldTransferLearning_UTILS().gwdf_test;
		int sNT_2 = 1;
		
		//init target
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_transfer = new GridWorldTransferLearning_UTILS().gwdf_test_L;
		int tNT_2 = 1;
		
		ArrayList<Integer> results_2 = testSupporter_S1_GW(gwdf_source, sNT_2, gwdf_transfer, tNT_2);
		
		System.out.println("Not Covered = " + results_2.get(0));
		assertEquals((int)results_2.get(0), tNT_2);
		System.out.println("Covered = " + results_2.get(1));
		assertEquals((int)results_2.get(1), sNT_2);
	}
	
	//|stateVar_source| > |stateVar_target|
	//not implemented yet as a feature of the tested class
	
//	//fire version
//	@Test
//	public void test_GW_2_F_M() {
//		//init source
//		//GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source = new GridWorldTransferLearning_UTILS().gwdf_fire;
//		//int sNT_2 = 1;
//		
//		//init target
//		//GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_transfer = new GridWorldTransferLearning_UTILS().gwdf_treasure;
//		//int tNT_2 = 1;
//		
//		//ArrayList<Integer> results_2 = testSupporter_S1_GW(gwdf_source, sNT_2, gwdf_transfer, tNT_2);
//		
//		//init source_1
//		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source_1 = new GridWorldTransferLearning_UTILS().gwdf_fire;
//		int sNT_2_1 = 1;
//		
//		//init source_2
//		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source_2 = new GridWorldTransferLearning_UTILS().gwdf_treasure;
//		int sNT_2_2 = 1;
//		
//		//init target
//		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_transfer = new GridWorldTransferLearning_UTILS().gwdf_fire;
//		int tNT_2 = 1;
//		
//		ArrayList<Integer> results_2 = testSupporter_S2_GW(gwdf_source_1, sNT_2_1, gwdf_source_2, sNT_2_2, gwdf_transfer, tNT_2);
//		
//		System.out.println("Not Covered = " + results_2.get(0));
//		assertEquals((int)results_2.get(0), tNT_2);
//		System.out.println("Covered = " + results_2.get(1));
//		assertEquals((int)results_2.get(1), sNT_2_1 + sNT_2_2);
//	}
//	
//	//pit version
//	@Test
//	public void test_GW_2_P_M() {
//		//init source
//		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source = new GridWorldTransferLearning_UTILS().gwdf_pit;
//		int sNT_2 = 1;
//		
//		//init target
//		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_transfer = new GridWorldTransferLearning_UTILS().gwdf_treasure;
//		int tNT_2 = 1;
//		
//		ArrayList<Integer> results_2 = testSupporter_S1_GW(gwdf_source, sNT_2, gwdf_transfer, tNT_2);
//		
//		System.out.println("Not Covered = " + results_2.get(0));
//		assertEquals((int)results_2.get(0), tNT_2);
//		System.out.println("Covered = " + results_2.get(1));
//		assertEquals((int)results_2.get(1), sNT_2);
//	}
	
	/** |stateVar_source| != |stateVar_target| &&  domain_source < domain_target */
	//|stateVar_source| < |stateVar_target| &&  domain_source < domain_target
	@Test
	public void test_GW_3() {
		//init source
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source = new GridWorldTransferLearning_UTILS().gwdf_treasure;
		int sNT_3 = 1;
		
		//init target
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_transfer = new GridWorldTransferLearning_UTILS().gwdf_test;
		int tNT_3 = 1;
		
		ArrayList<Integer> results_3 = testSupporter_S1_GW(gwdf_source, sNT_3, gwdf_transfer, tNT_3);
		
		System.out.println("Not Covered = " + results_3.get(0));
		assertEquals((int)results_3.get(0), tNT_3);
		System.out.println("Covered = " + results_3.get(1));
		assertEquals((int)results_3.get(1), sNT_3);
	}
	
	//fire version
	@Test
	public void test_GW_3_F() {
		//init source
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source = new GridWorldTransferLearning_UTILS().gwdf_fire;
		int sNT_3 = 1;
		
		//init target
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_transfer = new GridWorldTransferLearning_UTILS().gwdf_test;
		int tNT_3 = 1;
		
		ArrayList<Integer> results_3 = testSupporter_S1_GW(gwdf_source, sNT_3, gwdf_transfer, tNT_3);
		
		System.out.println("Not Covered = " + results_3.get(0));
		assertEquals((int)results_3.get(0), tNT_3);
		System.out.println("Covered = " + results_3.get(1));
		assertEquals((int)results_3.get(1), sNT_3);
	}
	
	//TODO
	//|stateVar_source| > |stateVar_target| &&  domain_source < domain_target
	
	/** |stateVar_source| != |stateVar_target| &&  domain_source > domain_target */
	
//	@Test
//	public void test_BD() {
//		
//		/** source_T = target_T */
//		
//		//init source
//		State sIS_0 = BlockDudeTransferLearning_UTILS.initialState_1;
//		int sNT_0 = 1;
//		
//		//init target
//		State tIS_0 = BlockDudeTransferLearning_UTILS.initialState_1;
//		int tNT_0 = 1;
//		
//		ArrayList<Integer> results_0 = testSupporter_S1(sIS_0, sNT_0, tIS_0, tNT_0);
//		
//		System.out.println("Not Covered = " + results_0.get(0));
//		assertEquals((int)results_0.get(0), tNT_0);
//		System.out.println("Covered = " + results_0.get(1));
//		assertEquals((int)results_0.get(1), sNT_0);
//		
//		/** domain_source < domain_target */
//		
//		//init source
//		State sIS_1 = BlockDudeTransferLearning_UTILS.initialState_1;
//		int sNT_1 = 1;
//		
//		//init target
//		State tIS_1 = BlockDudeTransferLearning_UTILS.initialState_1_L;
//		int tNT_1 = 1;
//		
//		ArrayList<Integer> results_1 = testSupporter_S1(sIS_1, sNT_1, tIS_1, tNT_1);
//		
//		System.out.println("Not Covered = " + results_1.get(0));
//		assertEquals((int)results_1.get(0), tNT_1);
//		System.out.println("Covered = " + results_1.get(1));
//		assertEquals((int)results_1.get(1), sNT_1);
//		
//		/** stateVar_source != stateVar_target */
//		
//		//init source
//		State sIS_2 = BlockDudeTransferLearning_UTILS.initialState_1;
//		int sNT_2 = 1;
//		
//		//init target
//		State tIS_2 = BlockDudeTransferLearning_UTILS.initialState_3;
//		int tNT_2 = 1;
//		
//		ArrayList<Integer> results_2 = testSupporter_S1(sIS_2, sNT_2, tIS_2, tNT_2);
//		
//		System.out.println("Not Covered = " + results_2.get(0));
//		assertEquals((int)results_2.get(0), tNT_2);
//		System.out.println("Covered = " + results_2.get(1));
//		assertEquals((int)results_2.get(1), sNT_2);
//		
//		/** stateVar_source != stateVar_target &&  domain_source < domain_target */
//		
//		//init source
//		State sIS_3 = BlockDudeTransferLearning_UTILS.initialState_1;
//		int sNT_3 = 1;
//		
//		//init target
//		State tIS_3 = BlockDudeTransferLearning_UTILS.initialState_2_L;
//		int tNT_3 = 1;
//		
//		ArrayList<Integer> results_3 = testSupporter_S1(sIS_3, sNT_3, tIS_3, tNT_3);
//		
//		System.out.println("Not Covered = " + results_3.get(0));
//		assertEquals((int)results_3.get(0), tNT_3);
//		System.out.println("Covered = " + results_2.get(1));
//		assertEquals((int)results_3.get(1), sNT_3);
//		
//		/** two different sources in cascade */
//		
//		//init source 1
//		State sIS_4_1 = BlockDudeTransferLearning_UTILS.initialState_1;
//		int sNT_4_1 = 1;
//		
//		//init source 2
//		State sIS_4_2 = BlockDudeTransferLearning_UTILS.initialState_2;
//		int sNT_4_2 = 1;
//		
//		//init target
//		State tIS_4 = BlockDudeTransferLearning_UTILS.initialState_2_L;
//		int tNT_4 = 1;
//		
//		ArrayList<Integer> results_4 = testSupporter_S2(sIS_4_1, sNT_4_1, sIS_4_2, sNT_4_2, tIS_4, tNT_4);
//		
//		System.out.println("Not Covered = " + results_4.get(0));
//		assertEquals((int)results_4.get(0), tNT_4);
//		System.out.println("Covered = " + results_4.get(1));
//		assertEquals((int)results_4.get(1), sNT_4_1 + sNT_4_2);
//		
//	}
	
	private ArrayList<Integer> testSupporter_S1(State sourceInitState, int sourceNTilings, State targetInitState, int targetNTilings){
		
		TLTileCodingFeatures sourceFeatures = BlockDudeTransferLearning_UTILS.CreateTransferLearningTCFeatures(sourceInitState, sourceNTilings);
		
		//create the features for this state
		sourceFeatures.features(sourceInitState);
		
		TLTileCodingFeatures targetFeatures = BlockDudeTransferLearning_UTILS.CreateTransferLearningTCFeatures(targetInitState, targetNTilings);
		
		//transfer
		sourceFeatures.transferKnowledge(targetFeatures, null);
		
		//create and take features for this state
		targetFeatures.features(targetInitState);
		
		ArrayList<Integer> toReturn = new ArrayList<Integer>();
		toReturn.add(targetFeatures.getTransferNotCovered());
		toReturn.add(targetFeatures.getTransferCovered());
		
		return toReturn;
	}
	
	private ArrayList<Integer> testSupporter_S2(State sourceInitState_1, int sourceNTilings_1,State sourceInitState_2, int sourceNTilings_2, State targetInitState, int targetNTilings){
		
		/** FIRST SOURCE */
		
		TLTileCodingFeatures sourceFeatures_1 = BlockDudeTransferLearning_UTILS.CreateTransferLearningTCFeatures(sourceInitState_1, sourceNTilings_1);
		
		//create the features for this state
		sourceFeatures_1.features(sourceInitState_1);
		
		/** SECOND SOURCE */
		
		TLTileCodingFeatures sourceFeatures_2 = BlockDudeTransferLearning_UTILS.CreateTransferLearningTCFeatures(sourceInitState_2, sourceNTilings_2);
		
		//transfer
		sourceFeatures_1.transferKnowledge(sourceFeatures_2, null);
		
		//create the features for this state
		sourceFeatures_2.features(sourceInitState_2);
		
		/** TARGET */
		
		TLTileCodingFeatures targetFeatures = BlockDudeTransferLearning_UTILS.CreateTransferLearningTCFeatures(targetInitState, targetNTilings);
		
		//transfer
		sourceFeatures_2.transferKnowledge(targetFeatures, null);
		
		//create and take features for this state
		targetFeatures.features(targetInitState);
		
		ArrayList<Integer> toReturn = new ArrayList<Integer>();
		toReturn.add(targetFeatures.getTransferNotCovered());
		toReturn.add(targetFeatures.getTransferCovered());
		
		return toReturn;
	}
	
	private ArrayList<Integer> testSupporter_S1_GW(GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source, int sourceNTilings, GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_target, int targetNTilings){
		
		TLTileCodingFeatures sourceFeatures = GridWorldTransferLearning_UTILS.CreateTransferLearningTCFeatures(gwdf_source.tag, sourceNTilings);
		
		//create the features for this state
		sourceFeatures.features(gwdf_source.generateInitialState());
		
		TLTileCodingFeatures targetFeatures = GridWorldTransferLearning_UTILS.CreateTransferLearningTCFeatures(gwdf_target.tag, targetNTilings);
		
		//transfer
		sourceFeatures.transferKnowledge(targetFeatures, null);
		
		//create and take features for this state
		targetFeatures.features(gwdf_target.generateInitialState());
		
		ArrayList<Integer> toReturn = new ArrayList<Integer>();
		toReturn.add(targetFeatures.getTransferNotCovered());
		toReturn.add(targetFeatures.getTransferCovered());
		
		return toReturn;
	}
	
	private ArrayList<Integer> testSupporter_S2_GW(GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source_1, int sourceNTilings_1, GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_source_2, int sourceNTilings_2, GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf_target, int targetNTilings){
		
		/** FIRST SOURCE */
		
		TLTileCodingFeatures sourceFeatures_1 = GridWorldTransferLearning_UTILS.CreateTransferLearningTCFeatures(gwdf_source_1.tag, sourceNTilings_1);
		
		//create the features for this state
		sourceFeatures_1.features(gwdf_source_1.generateInitialState());
		
		/** SECOND SOURCE */
		
		TLTileCodingFeatures sourceFeatures_2 = GridWorldTransferLearning_UTILS.CreateTransferLearningTCFeatures(gwdf_source_2.tag, sourceNTilings_2);
		
		//transfer
		sourceFeatures_1.transferKnowledge(sourceFeatures_2, null);
		
		//create the features for this state
		sourceFeatures_2.features(gwdf_source_2.generateInitialState());
		
		/** TARGET */
		
		TLTileCodingFeatures targetFeatures = GridWorldTransferLearning_UTILS.CreateTransferLearningTCFeatures(gwdf_target.tag, targetNTilings);
		
		//transfer
		sourceFeatures_2.transferKnowledge(targetFeatures, null);
		
		//create and take features for this state
		targetFeatures.features(gwdf_target.generateInitialState());
		
		ArrayList<Integer> toReturn = new ArrayList<Integer>();
		toReturn.add(targetFeatures.getTransferNotCovered());
		toReturn.add(targetFeatures.getTransferCovered());
		
		return toReturn;
	}

}
