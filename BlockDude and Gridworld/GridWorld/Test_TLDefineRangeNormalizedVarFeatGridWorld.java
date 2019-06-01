package GridWorld;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

import GridWorld.GridWorldTransferLearning_UTILS.GridWorldDomainFeatures;
import TransferLearning.VariableDomainFloatingNorm;
import burlap.behavior.functionapproximation.sparse.tilecoding.TilingArrangement;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.mdp.core.state.vardomain.VariableDomain;
import TransferLearning.TLTileCodingFeatures;

class Test_TLDefineRangeNormalizedVarFeatGridWorld {

	public int[][] MAP_test;
	public ArrayList<GridLocation> allFires_test;
	public ArrayList<GridLocation> allPits_test;
	public GridWorldTransferLearning_UTILS utils;
	
	@Test
	void test_relFire_IN() {
		this.init();
		
		allFires_test.add(new GridLocation(5,6,1,"fire1"));
		
		GridWorldDomainFeatures gwdf_test = utils.new GridWorldDomainFeatures(MAP_test, new GridAgent(5,5),new GridLocation(8,8,3,"treasure"), allFires_test, allPits_test, "");
		
		Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
		domains.put("treasure:x", new VariableDomainFloatingNorm(0,8));domains.put("treasure:y", new VariableDomainFloatingNorm(0,8));
		domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));	
		
		TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
		
		double[] obt = stateVariables.features(gwdf_test.generateInitialState());
		
		double[] exp = {0.3712, 0.3712, 0.495, 0.7425};
		
		System.out.println(obt[0] + " " + obt[1] + " " + obt[2] + " " + obt[3]);
		assertArrayEquals(exp, obt, 0.0001);
		
	}
	
	@Test
	void test_relFire_OUT() {
		this.init();
		
		allFires_test.add(new GridLocation(5,8,1,"fire1"));
		
		GridWorldDomainFeatures gwdf_test = utils.new GridWorldDomainFeatures(MAP_test, new GridAgent(5,5),new GridLocation(8,8,3,"treasure"), allFires_test, allPits_test, "");
		
		Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
		domains.put("treasure:x", new VariableDomainFloatingNorm(0,8));domains.put("treasure:y", new VariableDomainFloatingNorm(0,8));
		domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));	
		
		TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
		
		double[] obt = stateVariables.features(gwdf_test.generateInitialState());
		
		double[] exp = {0.3712, 0.3712, 1., 1.};
		
		System.out.println(obt[0] + " " + obt[1] + " " + obt[2] + " " + obt[3]);
		assertArrayEquals(exp, obt, 0.0001);
		
	}
	
	@Test
	void test_relPit_IN() {
		this.init();
		
		allPits_test.add(new GridLocation(5,4,1,"pit1"));
		
		GridWorldDomainFeatures gwdf_test = utils.new GridWorldDomainFeatures(MAP_test, new GridAgent(5,5),new GridLocation(8,8,3,"treasure"), allFires_test, allPits_test, "");
		
		Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
		domains.put("treasure:x", new VariableDomainFloatingNorm(0,8));domains.put("treasure:y", new VariableDomainFloatingNorm(0,8));
		domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
		
		TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
		
		double[] obt = stateVariables.features(gwdf_test.generateInitialState());
		
		double[] exp = {0.3712, 0.3712, 0.495, 0.2475};
		
		System.out.println(obt[0] + " " + obt[1] + " " + obt[2] + " " + obt[3]);
		assertArrayEquals(exp, obt, 0.0001);
		
	}
	
	@Test
	void test_relPit_OUT() {
		this.init();
		
		allPits_test.add(new GridLocation(5,2,1,"pit1"));
		
		GridWorldDomainFeatures gwdf_test = utils.new GridWorldDomainFeatures(MAP_test, new GridAgent(5,5),new GridLocation(8,8,3,"treasure"), allFires_test, allPits_test, "");
		
		Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
		domains.put("treasure:x", new VariableDomainFloatingNorm(0,8));domains.put("treasure:y", new VariableDomainFloatingNorm(0,8));
		domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
		
		TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
		
		double[] obt = stateVariables.features(gwdf_test.generateInitialState());
		
		double[] exp = {0.3712, 0.3712, 1., 1.};
		
		System.out.println(obt[0] + " " + obt[1] + " " + obt[2] + " " + obt[3]);
		assertArrayEquals(exp, obt, 0.0001);
	}
	
	@Test
	void test_relFire_relPit_1() {//1 fire and 1 pit possible to perceive
		this.init();
		
		allFires_test.add(new GridLocation(5,6,1,"fire1"));//in
		allFires_test.add(new GridLocation(5,8,1,"fire2"));//out
		
		allFires_test.add(new GridLocation(5,4,1,"pit1"));//in
		allFires_test.add(new GridLocation(5,2,1,"pit2"));//out
		
		GridWorldDomainFeatures gwdf_test = utils.new GridWorldDomainFeatures(MAP_test, new GridAgent(5,5),new GridLocation(8,8,3,"treasure"), allFires_test, allPits_test, "");
		
		Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
		domains.put("treasure:x", new VariableDomainFloatingNorm(0,8));domains.put("treasure:y", new VariableDomainFloatingNorm(0,8));
		domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));	
		domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
		
		TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
		
		double[] obt = stateVariables.features(gwdf_test.generateInitialState());
		
		double[] exp = {0.3712, 0.3712, 0.495, 0.7425, 0.495, 0.2475};
		
		System.out.println(obt[0] + " " + obt[1] + " " + obt[2] + " " + obt[3] + " " + obt[4] + " " + obt[5]);
		assertArrayEquals(exp, obt, 0.0001);
	}
	
	@Test
	void test_relFire_relPit_2() {//2 fires and 2 pits possible to perceive
		this.init();
		
		allFires_test.add(new GridLocation(5,6,1,"fire1"));//in
		allFires_test.add(new GridLocation(5,8,1,"fire2"));//out
		
		allFires_test.add(new GridLocation(5,4,1,"pit1"));//in
		allFires_test.add(new GridLocation(5,2,1,"pit2"));//out
		
		GridWorldDomainFeatures gwdf_test = utils.new GridWorldDomainFeatures(MAP_test, new GridAgent(5,5),new GridLocation(8,8,3,"treasure"), allFires_test, allPits_test, "");
		
		Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
		domains.put("treasure:x", new VariableDomainFloatingNorm(0,8));domains.put("treasure:y", new VariableDomainFloatingNorm(0,8));
		domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
		domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
		domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
		domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));	
		
		TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
		
		double[] obt = stateVariables.features(gwdf_test.generateInitialState());
		
		double[] exp = {0.3712, 0.3712, 0.495, 0.7425, 1., 1., 1., 1., 0.495, 0.2475};
		
		System.out.println(obt[0] + " " + obt[1] + " " + obt[2] + " " + obt[3] + " " + obt[4] + " " + obt[5] + " " + obt[6] + " " + obt[7] + " " + obt[8] + " " + obt[9]);
		assertArrayEquals(exp, obt, 0.0001);
	}
	
	@Test
	void test_relFire_relPit_compare_1() {//2 couples of fire and pit in diifferent absolute positions but identical relative ones
		this.init();
		
		allFires_test.add(new GridLocation(3,6,1,"fire1"));
		allFires_test.add(new GridLocation(8,6,1,"fire2"));
		
		allFires_test.add(new GridLocation(3,4,1,"pit1"));
		allFires_test.add(new GridLocation(8,4,1,"pit2"));
		
		GridWorldDomainFeatures gwdf_test_1 = utils.new GridWorldDomainFeatures(MAP_test, new GridAgent(3,5),new GridLocation(8,8,3,"treasure"), allFires_test, allPits_test, "");
		GridWorldDomainFeatures gwdf_test_2 = utils.new GridWorldDomainFeatures(MAP_test, new GridAgent(8,5),new GridLocation(8,8,3,"treasure"), allFires_test, allPits_test, "");
		
		Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
		//domains.put("treasure:x", new VariableDomainFloatingNorm(0,8));domains.put("treasure:y", new VariableDomainFloatingNorm(0,8));
		domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
		domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
		domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
		domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));	
		
		TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
		
		double[] obt_1 = stateVariables.features(gwdf_test_1.generateInitialState());
		double[] obt_2 = stateVariables.features(gwdf_test_2.generateInitialState());
		
		System.out.println(obt_1[0] + " " + obt_1[1] + " " + obt_1[2] + " " + obt_1[3] + " " + obt_1[4] + " " + obt_1[5] + " " + obt_1[6] + " " + obt_1[7]);
		System.out.println(obt_2[0] + " " + obt_2[1] + " " + obt_2[2] + " " + obt_2[3] + " " + obt_2[4] + " " + obt_2[5] + " " + obt_2[6] + " " + obt_2[7]);
		assertArrayEquals(obt_1, obt_2, 0.0001);
	}
	
	@Test
	void test_relFire_relPit_compare_2() {//2 couples of fire and pit in diifferent absolute positions but identical relative ones
		this.init();
		
		allFires_test.add(new GridLocation(3,6,1,"fire1"));
		allFires_test.add(new GridLocation(8,6,1,"fire2"));
		
		allFires_test.add(new GridLocation(3,4,1,"pit1"));
		allFires_test.add(new GridLocation(8,4,1,"pit2"));
		
		GridWorldDomainFeatures gwdf_test_1 = utils.new GridWorldDomainFeatures(MAP_test, new GridAgent(3,5),new GridLocation(8,8,3,"treasure"), allFires_test, allPits_test, "");
		GridWorldDomainFeatures gwdf_test_2 = utils.new GridWorldDomainFeatures(MAP_test, new GridAgent(8,5),new GridLocation(8,8,3,"treasure"), allFires_test, allPits_test, "");
		
		Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
		//domains.put("treasure:x", new VariableDomainFloatingNorm(0,8));domains.put("treasure:y", new VariableDomainFloatingNorm(0,8));
		domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));
		//domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
		domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
		//domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));	
		
		TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
		
		double[] obt_1 = stateVariables.features(gwdf_test_1.generateInitialState());
		double[] obt_2 = stateVariables.features(gwdf_test_2.generateInitialState());
		
		System.out.println(obt_1[0] + " " + obt_1[1] + " " + obt_1[2] + " " + obt_1[3]);
		System.out.println(obt_2[0] + " " + obt_2[1] + " " + obt_2[2] + " " + obt_2[3]);
		assertArrayEquals(obt_1, obt_2, 0.0001);
	}
	
	private void init() {
		this.utils = new GridWorldTransferLearning_UTILS();
		
		this.MAP_test = utils.MAP_treasure_XL;
		this.allFires_test = new ArrayList<GridLocation>();
		this.allPits_test = new ArrayList<GridLocation>();
		
	}
	
//	@Test
//	void test() {
//		GridWorldTransferLearning_UTILS utils = new GridWorldTransferLearning_UTILS();
//		
//		/** MAP_treasure_XL
//		 * 
//		 * O O O O O O O O O O O 
//		 * O                 T O
//		 * O                   O
//		 * O                   O
//		 * O   F F     F F     O
//		 * O                   O
//		 * O   P P     P P     O
//		 * O                   O
//		 * O                   O
//		 * O A                 O
//		 * O O O O O O O O O O O
//		 * */
//		
//		int[][] MAP_test = utils.MAP_treasure_XL;
//		
//		ArrayList<GridLocation> allFires_test = new ArrayList<GridLocation>(){{add(new GridLocation(2,6,1,"fire1"));
//																			add(new GridLocation(3,6,1,"fire2"));
//																			add(new GridLocation(6,6,1,"fire3"));
//																			add(new GridLocation(7,6,1,"fire4"));}};
//		ArrayList<GridLocation> allPits_test = new ArrayList<GridLocation>(){{add(new GridLocation(2,4,0,"pit1"));
//																			add(new GridLocation(3,4,0,"pit2"));
//																			add(new GridLocation(6,4,0,"pit3"));
//																			add(new GridLocation(7,4,0,"pit4"));}};
//		
//		GridWorldDomainFeatures gwdf_test = utils.new GridWorldDomainFeatures(MAP_test, new GridAgent(3,5),new GridLocation(8,8,3,"treasure"), allFires_test, allPits_test, "");
//		GridWorldDomainFeatures gwdf_test_2 = utils.new GridWorldDomainFeatures(MAP_test, new GridAgent(7,5),new GridLocation(8,8,3,"treasure"), allFires_test, allPits_test, "");
//		
//		//
//		Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
//		domains.put("treasure:x", new VariableDomainFloatingNorm(0,7));domains.put("treasure:y", new VariableDomainFloatingNorm(0,7));
//		domains.put("relFire1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire1:y", new VariableDomainFloatingNorm(-2,2));	
//		domains.put("relFire2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relFire2:y", new VariableDomainFloatingNorm(-2,2));
//		domains.put("relPit1:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit1:y", new VariableDomainFloatingNorm(-2,2));	
//		domains.put("relPit2:x", new VariableDomainFloatingNorm(-2,2));domains.put("relPit2:y", new VariableDomainFloatingNorm(-2,2));	
//		
//		TLDefineRangeNormalizedVarFeatGridWorld stateVariables = new TLDefineRangeNormalizedVarFeatGridWorld(domains);
//			
//		TLTileCodingFeatures tileCoding = new TLTileCodingFeatures(stateVariables);
//			
//		double[] tilingWidths = new double[domains.size()];
//			
//		tilingWidths[0] = 0.25;tilingWidths[1] = 0.25;
//		tilingWidths[2] = 0.25;tilingWidths[3] = 0.25;
//		tilingWidths[4] = 0.25;tilingWidths[5] = 0.25;
//		
//		tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, 8, TilingArrangement.UNIFORM);
//		
//		double[] obt = stateVariables.features(gwdf_test.generateInitialState());
//		double[] obt_2 = stateVariables.features(gwdf_test_2.generateInitialState());
//		
//		//double[] exp = {0.99, 0.99, 1., 1.};
//		assertArrayEquals(obt, obt_2, 0.00001);
//		//reimplementare ordinamento variabili
//		//scrivere codice per pit
//		//testare contro multipli fuochi
//		//testare contro multipli fuochi e pits
//	}

}
