package myProj;

import static org.junit.Assert.*;

import java.util.AbstractMap;
import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.state.BlockDudeAgent;
import burlap.domain.singleagent.blockdude.state.BlockDudeCell;
import burlap.domain.singleagent.blockdude.state.BlockDudeMap;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.mdp.core.state.vardomain.VariableDomain;

public class NormalizedVariableFeaturesBlockDudeTest {

	@Test
	public void testFeatures() {
		
		Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
		domains.put("block1:x", new VariableDomain(-7,4));
		domains.put("block1:y", new VariableDomain(-3,1));
		domains.put("block2:x", new VariableDomain(-7,4));
		domains.put("block2:y", new VariableDomain(-3,1));
		domains.put("block3:x", new VariableDomain(-7,4));
		domains.put("block3:y", new VariableDomain(-3,1));
		domains.put("exit:x", new VariableDomain(0,7));
		domains.put("exit:y", new VariableDomain(-3,0));
		NormalizedVariableFeaturesBlockDude BDFeatures = new NormalizedVariableFeaturesBlockDude(domains);
		
		//#########################TEST 0 normal 3 blocks map #######################################
		
		BlockDudeState initialState = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
				new BlockDudeMap(new int[][] {{1,1,1,1,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,1,1,1,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,1,1,1,0,0,0,0,0,0}}), 
				new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"), 
				new BlockDudeCell(3, 1, BlockDude.CLASS_BLOCK, "block1"),
				new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block2"),
				new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block3"));
		
		double[] expected0 = {9./11,3./4,10./11,3./4,1.,3./4,1.,1.};
		
		double[] out = BDFeatures.features(initialState);
		
		System.out.println("TEST 0\n");
		for(double d : out)
			System.out.println(d);
		
		assertArrayEquals(expected0, out, 0.00001);
		
		//##########################################################################################
		
		//#########################TEST 1 agent between movable blocks #############################
		//block1 and the agent have been inverted positions
		
		initialState = new BlockDudeState(new BlockDudeAgent(3, 1, 0, false), 
				new BlockDudeMap(new int[][] {{1,1,1,1,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,1,1,1,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,1,1,1,0,0,0,0,0,0}}), 
				new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"), 
				new BlockDudeCell(1, 1, BlockDude.CLASS_BLOCK, "block1"),
				new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block2"),
				new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block3"));
		
	
	
		double[] expected1 = {5./11,3./4,8./11,3./4,9./11,3./4,5./7,1.};
		
		out = BDFeatures.features(initialState);
		
		System.out.println("\nTEST 1\n");
		for(double d : out)
			System.out.println(d);
		
		assertArrayEquals(expected1, out, 0.00001);
		
		//#########################################################################################
		
		//###############################TEST 2 block1 on block3 ##################################
		//block1 has the maximum high a block could reach
				
		initialState = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
				new BlockDudeMap(new int[][] {{1,1,1,1,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,1,1,1,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,1,1,1,0,0,0,0,0,0}}), 
				new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"), 
				new BlockDudeCell(5, 2, BlockDude.CLASS_BLOCK, "block1"),
				new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block2"),
				new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block3"));
		
	
	
		double[] expected2 = {1.,1.,10./11,3./4,1.,3./4,1.,1.};;
		
		out = BDFeatures.features(initialState);
		
		System.out.println("\nTEST 2\n");
		for(double d : out)
			System.out.println(d);
		
		assertArrayEquals(expected2, out, 0.00001);
				
		//#########################################################################################
		
		//###############################TEST 3 agent on column ##################################
		//the agent is on the column of non movable blocks in the centre of the map
				
		initialState = new BlockDudeState(new BlockDudeAgent(6, 4, 0, false), 
				new BlockDudeMap(new int[][] {{1,1,1,1,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,1,1,1,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,1,1,1,0,0,0,0,0,0}}), 
				new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"), 
				new BlockDudeCell(3, 1, BlockDude.CLASS_BLOCK, "block1"),
				new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block2"),
				new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block3"));
		
	
	
		double[] expected3 = {4./11,0,5./11,0,6./11,0,2./7,0};;
		
		out = BDFeatures.features(initialState);
		
		System.out.println("\nTEST 3\n");
		for(double d : out)
			System.out.println(d);
		
		assertArrayEquals(expected3, out, 0.00001);
				
		//#########################################################################################
	}
}
