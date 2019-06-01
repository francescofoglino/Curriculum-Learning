package BlockDude;

import TransferLearning.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

import burlap.behavior.functionapproximation.dense.rbf.RBFFeatures;
import burlap.behavior.functionapproximation.sparse.tilecoding.TileCodingFeatures;
import burlap.behavior.functionapproximation.sparse.tilecoding.TilingArrangement;
import burlap.behavior.learningrate.ConstantLR;
import burlap.behavior.policy.GreedyDeterministicQPolicy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.QValue;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.state.BlockDudeAgent;
import burlap.domain.singleagent.blockdude.state.BlockDudeCell;
import burlap.domain.singleagent.blockdude.state.BlockDudeMap;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;

public class BlockDudeTransferLearning_UTILS {
	
	
	/** 
	 * ------------------------------------------------------------------------------------------
	 * ------------------------------------------ MAPS ------------------------------------------
	 * ------------------------------------------------------------------------------------------
	 */
	
	//***************************************** MAP 1 *******************************************
	
	/** MAP_1_min
	 * 
	 * 
	 * 
	 * 
	 * O     O
	 * O     O
	 * O     O
	 * O     O
	 * O A E O
	 * O O O O
	 * */	
	public final static int[][] MAP_1_min = {{1,1,1,1,1,1,0,0,0,0},
				  							{1,0,0,0,0,0,0,0,0,0},
				  							{1,0,0,0,0,0,0,0,0,0},
				  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_1_min = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
					new BlockDudeMap(MAP_1_min), 
					new BlockDudeCell(2, 1, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_1
	 * 
	 * 
	 * 
	 * 
	 * O                 O
	 * O                 O
	 * O                 O
	 * O                 O
	 * O A             E O
	 * O O O O O O O O O O
	 * */	
	public final static int[][] MAP_1 = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_1 = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
					new BlockDudeMap(MAP_1), 
					new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_1_L
	 * 
	 * 
	 * 
	 * 
	 * O                 		   O
	 * O                 		   O
	 * O                 		   O
	 * O                 		   O
	 * O A              		 E O
	 * O O O O O O O O O O O O O O O
	 * */	
	public final static int[][] MAP_1_L = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_1_L = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
					new BlockDudeMap(MAP_1_L), 
					new BlockDudeCell(13, 1, BlockDude.CLASS_EXIT, "exit"));
	
	//***************************************** MAP 2 *******************************************
	
	/** MAP_2_min
	 * 
	 * 
	 * 
	 * 
	 * O       O
	 * O       O
	 * O       O
	 * O       O
	 * O A O E O
	 * O O O O O
	 * */	
	public final static int[][] MAP_2_min = {{1,1,1,1,1,1,0,0,0,0},
				  							{1,0,0,0,0,0,0,0,0,0},
				  							{1,1,0,0,0,0,0,0,0,0},
				  							{1,0,0,0,0,0,0,0,0,0},
				  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_2_min = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
					new BlockDudeMap(MAP_2_min), 
					new BlockDudeCell(3, 1, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_2
	 * 
	 * 
	 * 
	 * 
	 * O                 O
	 * O                 O
	 * O                 O
	 * O                 O
	 * O A       O     E O
	 * O O O O O O O O O O
	 * */	
	public static final int[][] MAP_2 = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_2 = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_2), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_2_1
	 * 
	 * 
	 * 
	 * 
	 * O                 O
	 * O                 O
	 * O                 O
	 * O                 O
	 * O A O           E O
	 * O O O O O O O O O O
	 * */	
	public static final int[][] MAP_2_1 = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_2_1 = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_2_1), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_2_2
	 * 
	 * 
	 * 
	 * 
	 * O                 O
	 * O                 O
	 * O                 O
	 * O                 O
	 * O A   O         E O
	 * O O O O O O O O O O
	 * */	
	public static final int[][] MAP_2_2 = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_2_2 = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_2_2), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_2_3
	 * 
	 * 
	 * 
	 * 
	 * O                 O
	 * O                 O
	 * O                 O
	 * O                 O
	 * O A     O       E O
	 * O O O O O O O O O O
	 * */	
	public static final int[][] MAP_2_3 = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_2_3 = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_2_3), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_2_4
	 * 
	 * 
	 * 
	 * 
	 * O                 O
	 * O                 O
	 * O                 O
	 * O                 O
	 * O A         O   E O
	 * O O O O O O O O O O
	 * */	
	public static final int[][] MAP_2_4 = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_2_4 = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_2_4), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_2_5
	 * 
	 * 
	 * 
	 * 
	 * O                 O
	 * O                 O
	 * O                 O
	 * O                 O
	 * O A           O E O
	 * O O O O O O O O O O
	 * */	
	public static final int[][] MAP_2_5 = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_2_5 = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_2_5), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_2_L
	 * 
	 * 
	 * 
	 * 
	 * O                 		   O
	 * O                 		   O
	 * O                 		   O
	 * O                 		   O
	 * O A   O          		 E O
	 * O O O O O O O O O O O O O O O
	 * */	
	public final static int[][] MAP_2_L = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_2_L = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
					new BlockDudeMap(MAP_2_L), 
					new BlockDudeCell(13, 1, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_2_L2
	 * 
	 * 
	 * 
	 * 
	 * O                 		   O
	 * O                 		   O
	 * O                 		   O
	 * O                 		   O
	 * O A                   O   E O
	 * O O O O O O O O O O O O O O O
	 * */	
	public final static int[][] MAP_2_L2 = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_2_L2 = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
					new BlockDudeMap(MAP_2_L2), 
					new BlockDudeCell(13, 1, BlockDude.CLASS_EXIT, "exit"));
	
	//***************************************** MAP 3 *******************************************
	
	/** MAP_3_min
	 * 
	 * 
	 * 
	 * 
	 * O           O
	 * O           O
	 * O           O
	 * O       O   O
	 * O A X   O E O
	 * O O O O O O O
	 * */	
	
	public final static int[][] MAP_3_min = {{1,1,1,1,1,1,0,0,0,0},
				  							{1,0,0,0,0,0,0,0,0,0},
				  							{1,0,0,0,0,0,0,0,0,0},
				  							{1,0,0,0,0,0,0,0,0,0},
				  							{1,1,1,0,0,0,0,0,0,0},
				  							{1,0,0,0,0,0,0,0,0,0},
				  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_3_min = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
					new BlockDudeMap(MAP_3_min), 
					new BlockDudeCell(5, 1, BlockDude.CLASS_EXIT, "exit"),
					new BlockDudeCell(2, 1, BlockDude.CLASS_BLOCK, "block1"));
	
	/** MAP_3
	 * 
	 * 
	 * 
	 * 
	 * O                 O
	 * O                 O
	 * O                 O
	 * O           O     O
	 * O A   X     O   E O
	 * O O O O O O O O O O
	 * */	
	public static final int[][] MAP_3 = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_3 = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_3), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(3, 1, BlockDude.CLASS_BLOCK, "block1"));
	
	/** MAP_3_1
	 * 
	 * 
	 * 
	 * 
	 * O                 O
	 * O                 O
	 * O                 O
	 * O         O       O
	 * O A   X   O     E O
	 * O O O O O O O O O O
	 * */	
	public static final int[][] MAP_3_1 = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_3_1 = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_3_1), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(3, 1, BlockDude.CLASS_BLOCK, "block1"));
	
	/** MAP_3_2
	 * 
	 * 
	 * 
	 * 
	 * O                 O
	 * O                 O
	 * O                 O
	 * O             O   O
	 * O A     X     O E O
	 * O O O O O O O O O O
	 * */	
	public static final int[][] MAP_3_2 = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_3_2 = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_3_2), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block1"));
	
	/** MAP_3_L
	 * 
	 * 
	 * 
	 * 
	 * O                           O
	 * O                           O
	 * O                           O
	 * O                       O   O
	 * O A                 X   O E O
	 * O O O O O O O O O O O O O O O
	 * */	
	public static final int[][] MAP_3_L = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_3_L = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_3_L), 
			new BlockDudeCell(13, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(10, 1, BlockDude.CLASS_BLOCK, "block1"));
	
	/** MAP_3_L2
	 * 
	 * 
	 * 
	 * 
	 * O                           O
	 * O                           O
	 * O                           O
	 * O       O                   O
	 * O A X   O                 E O
	 * O O O O O O O O O O O O O O O
	 * */	
	public static final int[][] MAP_3_L2 = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_3_L2 = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_3_L2), 
			new BlockDudeCell(13, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(2, 1, BlockDude.CLASS_BLOCK, "block1"));
	
	//***************************************** MAP 4 *******************************************
	
	/** MAP_4              the minimum number of actions for solving this map is 9
	 * 
	 * 
	 * 
	 * 
	 * O                 O
	 * O                 O
	 * O           O     O
	 * O           O     O
	 * O A   X X X O   E O
	 * O O O O O O O O O O
	 * */	
	public static final int[][] MAP_4 = {{1,1,1,1,1,1,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,0,0,0,0,0,0,0,0,0},
			  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_4 = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_4), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(3, 1, BlockDude.CLASS_BLOCK, "block1"),
			new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block2"),
			new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block3"));
	
	/** optimal policy stated*/	
	static BlockDudeState map4_OPS_0 = initialState_4;
	
	static BlockDudeState map4_OPS_1 = new BlockDudeState(new BlockDudeAgent(2, 1, 1, false), 
			new BlockDudeMap(MAP_4), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(3, 1, BlockDude.CLASS_BLOCK, "block1"),
			new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block2"),
			new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block3"));
	
	static BlockDudeState map4_OPS_2 = new BlockDudeState(new BlockDudeAgent(2, 1, 1, true), 
			new BlockDudeMap(MAP_4), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(2, 2, BlockDude.CLASS_BLOCK, "block1"),
			new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block2"),
			new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block3"));
	
	static BlockDudeState map4_OPS_3 = new BlockDudeState(new BlockDudeAgent(3, 1, 1, true), 
			new BlockDudeMap(MAP_4), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(3, 2, BlockDude.CLASS_BLOCK, "block1"),
			new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block2"),
			new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block3"));
	
	static BlockDudeState map4_OPS_4 = new BlockDudeState(new BlockDudeAgent(4, 2, 1, true), 
			new BlockDudeMap(MAP_4), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(4, 3, BlockDude.CLASS_BLOCK, "block1"),
			new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block2"),
			new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block3"));
	
	static BlockDudeState map4_OPS_5 = new BlockDudeState(new BlockDudeAgent(4, 2, 1, false), 
			new BlockDudeMap(MAP_4), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(5, 2, BlockDude.CLASS_BLOCK, "block1"),
			new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block2"),
			new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block3"));
	
	static BlockDudeState map4_OPS_6 = new BlockDudeState(new BlockDudeAgent(5, 3, 1, false), 
			new BlockDudeMap(MAP_4), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(5, 2, BlockDude.CLASS_BLOCK, "block1"),
			new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block2"),
			new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block3"));
	
	static BlockDudeState map4_OPS_7 = new BlockDudeState(new BlockDudeAgent(6, 4, 1, false), 
			new BlockDudeMap(MAP_4), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(5, 2, BlockDude.CLASS_BLOCK, "block1"),
			new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block2"),
			new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block3"));
	
	static BlockDudeState map4_OPS_8 = new BlockDudeState(new BlockDudeAgent(7, 1, 1, false), 
			new BlockDudeMap(MAP_4), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(5, 2, BlockDude.CLASS_BLOCK, "block1"),
			new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block2"),
			new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block3"));
	
	//***************************************** MAP pit *******************************************
	
	/** MAP_pit_min
	 * 
	 * 
	 * O         O
	 * O         O       
	 * O X A   E O  
	 * O O O   O O 
	 *     O   O 
	 *     O O O
	 * */
	
	public static final int[][] MAP_pit_min = {{0,0,1,1,1,1,0,0},
											{0,0,1,0,0,0,0,0},
											{1,1,1,0,0,0,0,0},
											{1,0,0,0,0,0,0,0},
											{1,1,1,0,0,0,0,0},
											{0,0,1,1,1,1,0,0}};
	
	public final static BlockDudeState initialState_pit_min = new BlockDudeState(new BlockDudeAgent(2, 3, 0, false), 
																			new BlockDudeMap(MAP_pit_min), 
																			new BlockDudeCell(4, 3, BlockDude.CLASS_EXIT, "exit"),
																			new BlockDudeCell(1, 3, BlockDude.CLASS_BLOCK, "block1"));
	
	/** MAP_pit
	 * 
	 * 
	 * O                 O
	 * O                 O
	 * O X     A         O
	 * O O O O O       E O 
	 *         O   O O O O
	 *         O   O
	 * 		   O O O
	 * */	
	
	public static final int[][] MAP_pit = {{0,0,0,1,1,1,1,0,0},
											{0,0,0,1,0,0,0,0,0},
											{0,0,0,1,0,0,0,0,0},
											{0,0,0,1,0,0,0,0,0},
											{1,1,1,1,0,0,0,0,0},
											{1,0,0,0,0,0,0,0,0},
											{1,1,1,0,0,0,0,0,0},
											{0,0,1,0,0,0,0,0,0},
											{0,0,1,0,0,0,0,0,0},
											{0,0,1,1,1,1,1,0,0}};
	
	public final static BlockDudeState initialState_pit = new BlockDudeState(new BlockDudeAgent(4, 4, 0, false), 
																			new BlockDudeMap(MAP_pit), 
																			new BlockDudeCell(8, 3, BlockDude.CLASS_EXIT, "exit"),
																			new BlockDudeCell(1, 4, BlockDude.CLASS_BLOCK, "block1"));///////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	//*************************************** MAP reverse *****************************************
	
	/** MAP_reverse_min
	 * 
	 * 
	 * O     O
	 * O   A O
	 * O   O O
	 * O   E O
	 * O O O O
	 * */	
	
	public static final int[][] MAP_reverse_min = {{1,1,1,1,1,0,0},
												{1,0,0,0,0,0,0},
												{1,0,1,0,0,0,0},
												{1,1,1,1,1,0,0}};
	
	public final static BlockDudeState initialState_reverse_min = new BlockDudeState(new BlockDudeAgent(2, 3, 0, false), 
																				new BlockDudeMap(MAP_reverse_min), 
																				new BlockDudeCell(2, 1, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_reverse
	 * 
	 * 
	 * 
	 * O             O    
	 * O           A O
	 * O   O O O O O O   
	 * O             O
	 * O           E O
	 * O O O O O O O O
	 * */	
	
	public static final int[][] MAP_reverse = {{1,1,1,1,1,1,0,0,0},
											{1,0,0,0,0,0,0,0,0},
											{1,0,0,1,0,0,0,0,0},
											{1,0,0,1,0,0,0,0,0},
											{1,0,0,1,0,0,0,0,0},
											{1,0,0,1,0,0,0,0,0},
											{1,0,0,1,0,0,0,0,0},
											{1,1,1,1,1,1,0,0,0}};
	
	public final static BlockDudeState initialState_reverse = new BlockDudeState(new BlockDudeAgent(6, 4, 0, false), 
																			new BlockDudeMap(MAP_reverse), 
																			new BlockDudeCell(6, 1, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_reverse_trick
	 * 
	 * 
	 * 
	 * 
	 * O                 O    
	 * O E               O
	 * O O O O O     X   O   
	 * O           O O O O
	 * O A       O   
	 * O O O O O O 
	 * */	
	
	public static final int[][] MAP_reverse_trick = {{1,1,1,1,1,1,0,0,0,0},
													{1,0,0,1,0,0,0,0,0,0},
													{1,0,0,1,0,0,0,0,0,0},
													{1,0,0,1,0,0,0,0,0,0},
													{1,0,0,1,0,0,0,0,0,0},
													{1,1,0,0,0,0,0,0,0,0},
													{0,0,1,0,0,0,0,0,0,0},
													{0,0,1,0,0,0,0,0,0,0},
													{0,0,1,0,0,0,0,0,0,0},
													{0,0,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_reverse_trick = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
																				new BlockDudeMap(MAP_reverse_trick), 
																				new BlockDudeCell(1, 4, BlockDude.CLASS_EXIT, "exit"),
																				new BlockDudeCell(7, 3, BlockDude.CLASS_BLOCK, "block1"));
	
	//*************************************** MAP blocks ******************************************
	
	/** MAP_blocks_short
	 * 
	 * 
	 * 
	 * O                         O
	 * O     O                   O
	 * O     O               X   O
	 * O     O     X   A     X X O
	 * O E O O O O O O O O O O O O
	 * O O O
	 * */	
	
	public static final int[][] MAP_blocks_short = {{1,1,1,1,1,1,0,0,0},
													{1,0,0,0,0,0,0,0,0},
													{1,1,0,0,0,0,0,0,0},
													{0,1,1,1,1,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,1,1,1,1,0,0,0}};
	
	public final static BlockDudeState initialState_blocks_short = new BlockDudeState(new BlockDudeAgent(8, 2, 0, false), 
																					new BlockDudeMap(MAP_blocks_short), 
																					new BlockDudeCell(1, 1, BlockDude.CLASS_EXIT, "exit"),
																					new BlockDudeCell(6, 2, BlockDude.CLASS_BLOCK, "block1"),
																					new BlockDudeCell(11, 2, BlockDude.CLASS_BLOCK, "block2"),
																					new BlockDudeCell(11, 3, BlockDude.CLASS_BLOCK, "block3"),
																					new BlockDudeCell(12, 2, BlockDude.CLASS_BLOCK, "block4"));
	
	/** MAP_blocks_short_2
	 * 
	 * 
	 * 
	 * O                         O
	 * O                   O     O
	 * O   X               O     O
	 * O X X     A   X     O     O
	 * O O O O O O O O O O O O E O
	 *                       O O O
	 * */	
	
	public static final int[][] MAP_blocks_short_2 = {{0,1,1,1,1,1,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,0,0,0,0,0,0,0},
													{0,1,1,1,1,0,0,0,0},
													{1,1,0,0,0,0,0,0,0},
													{1,0,0,0,0,0,0,0,0},
													{1,1,1,1,1,1,0,0,0}};
	
	public final static BlockDudeState initialState_blocks_short_2 = new BlockDudeState(new BlockDudeAgent(5, 2, 0, false), 
																					new BlockDudeMap(MAP_blocks_short_2), 
																					new BlockDudeCell(12, 1, BlockDude.CLASS_EXIT, "exit"),
																					new BlockDudeCell(1, 2, BlockDude.CLASS_BLOCK, "block1"),
																					new BlockDudeCell(2, 2, BlockDude.CLASS_BLOCK, "block2"),
																					new BlockDudeCell(2, 3, BlockDude.CLASS_BLOCK, "block3"),
																					new BlockDudeCell(7, 2, BlockDude.CLASS_BLOCK, "block4"));
	
	/** MAP_blocks_short_3
	 * 
	 * 
	 * 
	 * O                         O
	 * O                   O     O
	 * O   X               O     O
	 * O X X     A   X     O   E O
	 * O O O O O O O O O O O O O O
	 *                       
	 * */	
	
	public static final int[][] MAP_blocks_short_3 = {{1,1,1,1,1,0,0,0},
													{1,0,0,0,0,0,0,0},
													{1,0,0,0,0,0,0,0},
													{1,0,0,0,0,0,0,0},
													{1,0,0,0,0,0,0,0},
													{1,0,0,0,0,0,0,0},
													{1,0,0,0,0,0,0,0},
													{1,0,0,0,0,0,0,0},
													{1,0,0,0,0,0,0,0},
													{1,0,0,0,0,0,0,0},
													{1,1,1,1,0,0,0,0},
													{1,0,0,0,0,0,0,0},
													{1,0,0,0,0,0,0,0},
													{1,1,1,1,1,0,0,0}};
	
	public final static BlockDudeState initialState_blocks_short_3 = new BlockDudeState(new BlockDudeAgent(5, 1, 0, false), 
																					new BlockDudeMap(MAP_blocks_short_3), 
																					new BlockDudeCell(12, 1, BlockDude.CLASS_EXIT, "exit"),
																					new BlockDudeCell(1, 1, BlockDude.CLASS_BLOCK, "block1"),
																					new BlockDudeCell(2, 1, BlockDude.CLASS_BLOCK, "block2"),
																					new BlockDudeCell(2, 2, BlockDude.CLASS_BLOCK, "block3"),
																					new BlockDudeCell(7, 1, BlockDude.CLASS_BLOCK, "block4"));
	
	
	/** MAP_bloks_pit
	 * 
	 * 
	 * 
	 * O                                     O
	 * O                 O                   O
	 * O                 O               X   O
	 * O E               O     X   A     X X O
	 * O O O   O O O O O O O O O O O O O O O O
	 *     O   O
	 *     O   O
	 *     O O O
	 * */	
	
	public static final int[][] MAP_bloks_pit = {{0,0,0,1,1,1,1,1,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{1,1,1,1,0,0,0,0,0,0,0},
												{1,0,0,0,0,0,0,0,0,0,0},
												{1,1,1,1,0,0,0,0,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{0,0,0,1,1,1,1,0,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{0,0,0,1,0,0,0,0,0,0,0},
												{0,0,0,1,1,1,1,1,0,0,0}};
	
	public final static BlockDudeState initialState_blocks_pit = new BlockDudeState(new BlockDudeAgent(14, 4, 0, false), 
																					new BlockDudeMap(MAP_bloks_pit), 
																					new BlockDudeCell(1, 4, BlockDude.CLASS_EXIT, "exit"),
																					new BlockDudeCell(12, 4, BlockDude.CLASS_BLOCK, "block1"),
																					new BlockDudeCell(17, 4, BlockDude.CLASS_BLOCK, "block2"),
																					new BlockDudeCell(17, 5, BlockDude.CLASS_BLOCK, "block3"),
																					new BlockDudeCell(18, 4, BlockDude.CLASS_BLOCK, "block4"));
	
	//***************************************** MAP step *******************************************
	
	/** MAP_step_min
	 * 
	 * 
	 * 
	 * 
	 * O     O
	 * O   E O
	 * O A O O
	 * O O O O
	 * */	
	
	public static final int[][] MAP_step_min = {{1,1,1,1,0,0,0,0},
												{1,0,0,0,0,0,0,0},
												{1,1,0,0,0,0,0,0},
												{1,1,1,1,1,0,0,0}};
	
	public final static BlockDudeState initialState_step_min = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
																				new BlockDudeMap(MAP_step_min), 
																				new BlockDudeCell(2, 2, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_step
	 * 
	 * 
	 * 
	 * 
	 * O                 O
	 * O               E O
	 * O       A       O O
	 * O O O O O O O O O O
	 * */	
	
	public static final int[][] MAP_step = {{1,1,1,1,0,0,0,0},
											{1,0,0,0,0,0,0,0},
											{1,0,0,0,0,0,0,0},
											{1,0,0,0,0,0,0,0},
											{1,0,0,0,0,0,0,0},
											{1,0,0,0,0,0,0,0},
											{1,0,0,0,0,0,0,0},
											{1,0,0,0,0,0,0,0},
											{1,1,0,0,0,0,0,0},
											{1,1,1,1,1,0,0,0}};
	
	public final static BlockDudeState initialState_step = new BlockDudeState(new BlockDudeAgent(4, 1, 0, false), 
																			new BlockDudeMap(MAP_step), 
																			new BlockDudeCell(8, 2, BlockDude.CLASS_EXIT, "exit"));
	
	//***************************************** MAP rndm *******************************************
	
	/** MAP_reuse 
	 * 
	 * 
	 * O                               O
	 * O                               O
	 * O E                             O
	 * O O                             O
	 *   O                 O     X     O
	 *   O                 O X   X X A O
	 *   O O O O       O O O O O O O O O
	 *   	   O     X O
	 *         O O O O O
	 * */	
	
	public static final int[][] MAP_reuse = {{0,0,0,0,0,1,1,1,1,0,0},
											{0,0,1,1,1,1,0,0,0,0,0},
											{0,0,1,0,0,0,0,0,0,0,0},
											{0,0,1,0,0,0,0,0,0,0,0},
											{1,1,1,0,0,0,0,0,0,0,0},
											{1,0,0,0,0,0,0,0,0,0,0},
											{1,0,0,0,0,0,0,0,0,0,0},
											{1,0,0,0,0,0,0,0,0,0,0},
											{1,1,1,0,0,0,0,0,0,0,0},
											{0,0,1,0,0,0,0,0,0,0,0},
											{0,0,1,1,1,0,0,0,0,0,0},
											{0,0,1,0,0,0,0,0,0,0,0},
											{0,0,1,0,0,0,0,0,0,0,0},
											{0,0,1,0,0,0,0,0,0,0,0},
											{0,0,1,0,0,0,0,0,0,0,0},
											{0,0,1,0,0,0,0,0,0,0,0},
											{0,0,1,1,1,1,1,1,1,0,0}};
	
	public final static BlockDudeState initialState_reuse = new BlockDudeState(new BlockDudeAgent(15, 3, 0, false), 
																					new BlockDudeMap(MAP_reuse), 
																					new BlockDudeCell(1, 6, BlockDude.CLASS_EXIT, "exit"),
																					new BlockDudeCell(7, 1, BlockDude.CLASS_BLOCK, "block1"),
																					new BlockDudeCell(11, 3, BlockDude.CLASS_BLOCK, "block2"),
																					new BlockDudeCell(13, 3, BlockDude.CLASS_BLOCK, "block3"),
																					new BlockDudeCell(13, 4, BlockDude.CLASS_BLOCK, "block4"),
																					new BlockDudeCell(14, 3, BlockDude.CLASS_BLOCK, "block5"));
	
	
	
	/** MAP_useless_bc
	 * 
	 * O                       O
	 * O                       O
	 * O     O   A   X       E O
	 * O O O O O O O O O O O O O
	 * */	
	
	public static final int[][] MAP_useless_bc = {{1,1,1,1,0},
												{1,0,0,0,0},
												{1,0,0,0,0},
												{1,1,0,0,0},
												{1,0,0,0,0},
												{1,0,0,0,0},
												{1,0,0,0,0},
												{1,0,0,0,0},
												{1,0,0,0,0},
												{1,0,0,0,0},
												{1,0,0,0,0},
												{1,0,0,0,0},
												{1,1,1,1,0}};	
	
	public final static BlockDudeState initialState_useless_bc = new BlockDudeState(new BlockDudeAgent(5, 1, 0, false), 
																				new BlockDudeMap(MAP_useless_bc), 
																				new BlockDudeCell(11, 1, BlockDude.CLASS_EXIT, "exit"),
																				new BlockDudeCell(7, 1, BlockDude.CLASS_BLOCK, "block1"));
																		
	/****************************************************** FOR CURRICULA ****************************************************/
	
	/** MAP_2_hard
	 * 
	 * 
	 * 
	 * 
	 * O                 O
	 * O                 O
	 * O         O       O
	 * O       O O       O
	 * O A   O O O     E O
	 * O O O O O O O O O O
	 * */	
	public static final int[][] MAP_2_hard = {{1,1,1,1,1,1,0,0,0,0},
			  								{1,0,0,0,0,0,0,0,0,0},
			  								{1,0,0,0,0,0,0,0,0,0},
			  								{1,1,0,0,0,0,0,0,0,0},
			  								{1,1,1,0,0,0,0,0,0,0},
			  								{1,1,1,1,0,0,0,0,0,0},
			  								{1,0,0,0,0,0,0,0,0,0},
			  								{1,0,0,0,0,0,0,0,0,0},
			  								{1,0,0,0,0,0,0,0,0,0},
			  								{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_2_hard = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_2_hard), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"));
	
	public final static BlockDudeState initialState_2_hard_firstStep = new BlockDudeState(new BlockDudeAgent(2, 1, 0, false), 
			new BlockDudeMap(MAP_2_hard), 
			new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"));
	
	/** MAP_CURRICULUM_TARGET_min
	 * 
	 * 
	 * 
	 * 
	 * O                           O
	 * O                           O
	 * O                           O
	 * O       O                   O
	 * O A X   O             O   E O
	 * O O O O O O O O O O O O O O O
	 * */	
	public static final int[][] MAP_CURRICULUM_TARGET_min = {{1,1,1,1,1,1,0,0,0,0},
								  							{1,0,0,0,0,0,0,0,0,0},
								  							{1,0,0,0,0,0,0,0,0,0},
								  							{1,0,0,0,0,0,0,0,0,0},
								  							{1,1,1,0,0,0,0,0,0,0},
								  							{1,0,0,0,0,0,0,0,0,0},
								  							{1,0,0,0,0,0,0,0,0,0},
								  							{1,0,0,0,0,0,0,0,0,0},
								  							{1,0,0,0,0,0,0,0,0,0},
								  							{1,0,0,0,0,0,0,0,0,0},
								  							{1,0,0,0,0,0,0,0,0,0},
								  							{1,0,0,0,0,0,0,0,0,0},
								  							{1,0,0,0,0,0,0,0,0,0},
								  							{1,0,0,0,0,0,0,0,0,0},
								  							{1,1,1,1,1,1,0,0,0,0}};
	
	public final static BlockDudeState initialState_CURRICULUM_TARGET_min = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
			new BlockDudeMap(MAP_CURRICULUM_TARGET_min), 
			new BlockDudeCell(13, 1, BlockDude.CLASS_EXIT, "exit"),
			new BlockDudeCell(2, 1, BlockDude.CLASS_BLOCK, "block1"));	
	
	/** MAP_CURRICULUM_TARGET
	 * 
	 * 
	 * 
	 * 
	 * O                 		             O
	 * O                 		             O
	 * O                 		     O       O
	 * O             O   		     O       O
	 * O A     X X   O   X   O	 X   O     E O
	 * O O O O O O O O O O O O O O O O O O O O
	 * */	
	
	public final static int[][] MAP_CURRICULUM_TARGET = {{1,1,1,1,1,1,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,1,1,0,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,1,0,0,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,1,1,1,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,0,0,0,0,0,0,0,0,0},
														{1,1,1,1,1,1,0,0,0,0}};

	public final static BlockDudeState initialState_CURRICULUM_TARGET = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
	new BlockDudeMap(MAP_CURRICULUM_TARGET), 
	new BlockDudeCell(18, 1, BlockDude.CLASS_EXIT, "exit"),
	new BlockDudeCell(4, 1, BlockDude.CLASS_BLOCK, "block1"),
	new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block2"),
	new BlockDudeCell(9, 1, BlockDude.CLASS_BLOCK, "block3"),
	new BlockDudeCell(13, 1, BlockDude.CLASS_BLOCK, "block4"));
	
	/** 
	 * ------------------------------------------------------------------------------------------
	 * ------------------------------------------ END -------------------------------------------
	 * ------------------------------------------------------------------------------------------
	 */
	
	
	/** Function for writing the action-values relative to the specified state s on file*/
	public static void WriteActionValues(PrintWriter writer, QProvider qp, State s, long ep){
		writer.print(ep + " ");
		List<QValue> sQValues = qp.qValues(s);
		for(ListIterator<QValue> itr = sQValues.listIterator(); itr.hasNext() ;){
			QValue currentQValue = itr.next();
			double actionValue = currentQValue.q;
			
			writer.print(actionValue+ " ");
		}
		writer.println();
	}
	
	public static boolean bestVsExpectedAction(QProvider qp, State s, String expected){
		
		List<QValue> sQValues = qp.qValues(s);
		
		double maxVal = sQValues.get(0).q;
		String maxStr = sQValues.get(0).a.toString();
		
		for(ListIterator<QValue> itr = sQValues.listIterator(); itr.hasNext() ;){
			QValue currentQValue = itr.next();
			double actionValue = currentQValue.q;
			
			if(actionValue > maxVal){
				maxVal = actionValue;
				maxStr = currentQValue.a.toString();
			}
		}
		
		return expected.equals(maxStr);
	}
	
	/** Function for performing and saving the result of the execution of the greedy policy over the learnt Action-Value Function
	 *  @param episode is: <0 when we just want to print the execution of the episode, >0 when we want to have the reward file*/
	public static void PerformLearntPolicy(GradientDescentSarsaLam agentRef, SimulatedEnvironment environment, String outputPath, String name, long episode){
		
		//modify the agent for fully not learning behaviours 
		agentRef.setLearningPolicy(new GreedyDeterministicQPolicy(agentRef));
		agentRef.setLearningRate(new ConstantLR(0.0)); // alpha=0
		
		Episode e = agentRef.runLearningEpisode(environment,50);//object for storing the episode at each loop
		
		if(episode >= 0){
			BufferedWriter buff = null;
			try {
				FileWriter osPE = new FileWriter(outputPath+name, true);
				buff = new BufferedWriter(osPE);
				buff.write("\n" + episode + " " + e.discountedReturn(1) + " " + agentRef.getLastNumSteps());
				buff.close();
			} catch (IOException e1) {
				System.err.println("Error: " + e1.getMessage());
			}
		}else	
			e.write(outputPath + "_" + name);
		
		environment.resetEnvironment();
		
	}
	
	public static TLTileCodingFeatures CreateTransferLearningTCFeatures(State s0, int numTilings){
		
		TLTileCodingFeatures tileCoding = null;
		
		if(s0 == initialState_1){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//Former method = CreateDomain(1, "10x10");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,7));domains.put("exit:y", new VariableDomainFloatingNorm(-1,0));//This second is -1 0 cause it is the minimum interval possible even if the correct one would be 0 0
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.1; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			
			tileCoding.addTilingsForDimensionsAndWidths(new boolean [] {true, true, true, true}, tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
			/** Minimal implementation for the current map
			
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//Former method = CreateDomain(1, "10x10");
			domains.put("exit:x", new VariableDomain(0,7));			
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.3;
			
			tileCoding.addTilingsForDimensionsAndWidths(new boolean [] {true}, tilingWidths, numTilings, TilingArrangement.UNIFORM);
			*/
		}else if(s0 == initialState_3){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,7));domains.put("exit:y", new VariableDomainFloatingNorm(-2,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-4,4));domains.put("block1:y",new VariableDomainFloatingNorm(-2,1));
			domains.put("column1:x",new VariableDomainFloatingNorm(-2,5));domains.put("column1:y",new VariableDomainFloatingNorm(-1,1));
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.1; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.1; tilingWidths[5] = 0.5;
			tilingWidths[6] = 0.1; tilingWidths[7] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_3_1){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,7));domains.put("exit:y", new VariableDomainFloatingNorm(-2,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-7,3));domains.put("block1:y",new VariableDomainFloatingNorm(-2,1));
			domains.put("column1:x",new VariableDomainFloatingNorm(-3,4));domains.put("column1:y",new VariableDomainFloatingNorm(-1,1));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.1; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.1; tilingWidths[5] = 0.5;
			tilingWidths[6] = 0.1; tilingWidths[7] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_3_2){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,7));domains.put("exit:y", new VariableDomainFloatingNorm(-2,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-7,5));domains.put("block1:y",new VariableDomainFloatingNorm(-2,1));
			domains.put("column1:x",new VariableDomainFloatingNorm(-1,6));domains.put("column1:y",new VariableDomainFloatingNorm(-1,1));
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.1; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.1; tilingWidths[5] = 0.5;
			tilingWidths[6] = 0.1; tilingWidths[7] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_4){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,7));domains.put("exit:y", new VariableDomainFloatingNorm(-3,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-4,4));domains.put("block1:y",new VariableDomainFloatingNorm(-3,1));
			domains.put("block2:x",new VariableDomainFloatingNorm(-4,4));domains.put("block2:y",new VariableDomainFloatingNorm(-3,1));
			domains.put("block3:x",new VariableDomainFloatingNorm(-4,4));domains.put("block3:y",new VariableDomainFloatingNorm(-3,1));
			domains.put("column1:x",new VariableDomainFloatingNorm(-2,5));domains.put("column1:y",new VariableDomainFloatingNorm(-1,2));
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.1; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.1; tilingWidths[5] = 0.3;
			tilingWidths[6] = 0.1; tilingWidths[7] = 0.3;
			tilingWidths[8] = 0.1; tilingWidths[9] = 0.3;
			tilingWidths[10] = 0.1; tilingWidths[11] = 0.3;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_1_L){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//CreateDomain(1 , "large");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,12));domains.put("exit:y", new VariableDomainFloatingNorm(-1,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 1./6; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_3_L){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//CreateDomain(2, "large");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,12));domains.put("exit:y", new VariableDomainFloatingNorm(-2,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-10,10));domains.put("block1:y",new VariableDomainFloatingNorm(-2,1));
			domains.put("column1:x",new VariableDomainFloatingNorm(-1,11));domains.put("column1:y",new VariableDomainFloatingNorm(-1,1));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 1./6; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 1./10; tilingWidths[5] = 0.3;
			tilingWidths[6] = 1./6; tilingWidths[7] = 0.3;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_2_L){//took care
			Map<Object, VariableDomain> domains =  new HashMap<Object, VariableDomain>();//CreateDomain(1 , "try");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,12));domains.put("exit:y", new VariableDomainFloatingNorm(-1,0));//*
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));//*
			domains.put("column1:x", new VariableDomainFloatingNorm(-10,2));domains.put("column1:y", new VariableDomainFloatingNorm(-1,0));//*
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 1./6; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 1./6; tilingWidths[5] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			//tileCoding.addTilingsForDimensionsAndWidths(new boolean [] {true, true, true, true, false, false}, tilingWidths, 4, TilingArrangement.UNIFORM);
			//tileCoding.addTilingsForDimensionsAndWidths(new boolean [] {false, false, false, false, true, true}, tilingWidths, 4, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_2_L2){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//CreateDomain(2 , "try");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,12));domains.put("exit:y", new VariableDomainFloatingNorm(-1,0));//*
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));//*
			domains.put("column1:x", new VariableDomainFloatingNorm(-2,10));domains.put("column1:y", new VariableDomainFloatingNorm(-1,0));//*
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 1./6; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 1/.6; tilingWidths[5] = 0.5;
			
			tileCoding.addTilingsForDimensionsAndWidths(new boolean [] {true, true, true, true, true, true}, tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_3_L2){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//CreateDomain(100, "try");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,12));domains.put("exit:y", new VariableDomainFloatingNorm(-2,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-10,2));domains.put("block1:y",new VariableDomainFloatingNorm(-2,1));
			domains.put("column1:x",new VariableDomainFloatingNorm(-9,3));domains.put("column1:y",new VariableDomainFloatingNorm(-1,1));
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 1./6; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 1./10; tilingWidths[5] = 0.3;
			tilingWidths[6] = 1./6; tilingWidths[7] = 0.3;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_2){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//CreateDomain(0, "column");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,7));domains.put("exit:y", new VariableDomainFloatingNorm(-1,0));//*
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));//*
			domains.put("column1:x", new VariableDomainFloatingNorm(-3,4));domains.put("column1:y", new VariableDomainFloatingNorm(-1,0));//*
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.3; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.3; tilingWidths[5] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_2_1){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//CreateDomain(1, "column");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,7));domains.put("exit:y", new VariableDomainFloatingNorm(-1,0));//*
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));//*
			domains.put("column1:x", new VariableDomainFloatingNorm(-6,1));domains.put("column1:y", new VariableDomainFloatingNorm(-1,0));//*
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.3; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.3; tilingWidths[5] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_2_2){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//CreateDomain(2, "column");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,7));domains.put("exit:y", new VariableDomainFloatingNorm(-1,0));//*
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));//*
			domains.put("column1:x", new VariableDomainFloatingNorm(-5,2));domains.put("column1:y", new VariableDomainFloatingNorm(-1,0));//*
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.3; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.3; tilingWidths[5] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_2_3){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//CreateDomain(3, "column");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,7));domains.put("exit:y", new VariableDomainFloatingNorm(-1,0));//*
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));//*
			domains.put("column1:x", new VariableDomainFloatingNorm(-4,3));domains.put("column1:y", new VariableDomainFloatingNorm(-1,0));//*
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.3; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.3; tilingWidths[5] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_2_4){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//CreateDomain(4, "column");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,7));domains.put("exit:y", new VariableDomainFloatingNorm(-1,0));//*
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));//*
			domains.put("column1:x", new VariableDomainFloatingNorm(-2,5));domains.put("column1:y", new VariableDomainFloatingNorm(-1,0));//*
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.3; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.3; tilingWidths[5] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_2_5){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//CreateDomain(5, "column");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,7));domains.put("exit:y", new VariableDomainFloatingNorm(-1,0));//*
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));//*
			domains.put("column1:x", new VariableDomainFloatingNorm(-1,6));domains.put("column1:y", new VariableDomainFloatingNorm(-1,0));//*
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.3; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.3; tilingWidths[5] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_1_min){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//CreateDomain(0, "atomic");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,1));domains.put("exit:y", new VariableDomainFloatingNorm(-1,0));//*
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));//*
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			
			tileCoding.addTilingsForDimensionsAndWidths(new boolean [] {true, true, true, true}, tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_2_min){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();//CreateDomain(1, "atomic");
			domains.put("exit:x", new VariableDomainFloatingNorm(0,2));domains.put("exit:y", new VariableDomainFloatingNorm(-1,0));//*
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));//*
			domains.put("column1:x", new VariableDomainFloatingNorm(-1,1));domains.put("column1:y", new VariableDomainFloatingNorm(-1,0));//*
			
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.3; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.3; tilingWidths[5] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_3_min){//OK for curriculum
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,4));domains.put("exit:y", new VariableDomainFloatingNorm(-2,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-2,2));domains.put("block1:y",new VariableDomainFloatingNorm(-2,1));
			domains.put("column1:x",new VariableDomainFloatingNorm(-1,3));domains.put("column1:y",new VariableDomainFloatingNorm(-1,1));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.2; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.2; tilingWidths[5] = 0.5;
			tilingWidths[6] = 0.2; tilingWidths[7] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_pit_min){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,3));domains.put("exit:y", new VariableDomainFloatingNorm(-1,2));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-2,2));domains.put("block1:y",new VariableDomainFloatingNorm(-2,2));
			domains.put("column1:x",new VariableDomainFloatingNorm(-2,1));domains.put("column1:y",new VariableDomainFloatingNorm(-1,2));
			domains.put("column2:x",new VariableDomainFloatingNorm(0,3));domains.put("column2:y",new VariableDomainFloatingNorm(-1,2));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.3;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.25; tilingWidths[5] = 0.5;
			tilingWidths[6] = 0.5; tilingWidths[7] = 0.5;
			tilingWidths[8] = 0.5; tilingWidths[9] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_pit){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,7));domains.put("exit:y", new VariableDomainFloatingNorm(-2,2));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-4,4));domains.put("block1:y",new VariableDomainFloatingNorm(-3,3));
			domains.put("column1:x",new VariableDomainFloatingNorm(-4,3));domains.put("column1:y",new VariableDomainFloatingNorm(-3,2));
			domains.put("column2:x",new VariableDomainFloatingNorm(-2,5));domains.put("column2:y",new VariableDomainFloatingNorm(-2,2));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.25; tilingWidths[1] = 0.3;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.2; tilingWidths[5] = 0.3;
			tilingWidths[6] = 0.3; tilingWidths[7] = 0.3;
			tilingWidths[8] = 0.3; tilingWidths[9] = 0.3;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_reverse_min){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,1));domains.put("exit:y", new VariableDomainFloatingNorm(-2,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("column1:x",new VariableDomainFloatingNorm(0,1));domains.put("column1:y",new VariableDomainFloatingNorm(-1,1));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.5; tilingWidths[5] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_reverse){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,5));domains.put("exit:y", new VariableDomainFloatingNorm(-3,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("column1:x",new VariableDomainFloatingNorm(-4,1));domains.put("column1:y",new VariableDomainFloatingNorm(-1,2));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.3; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.3; tilingWidths[5] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_reverse_trick){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(-7,0));domains.put("exit:y", new VariableDomainFloatingNorm(0,3));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-7,7));domains.put("block1:y",new VariableDomainFloatingNorm(-2,2));
			domains.put("column1:x",new VariableDomainFloatingNorm(-4,3));domains.put("column1:y",new VariableDomainFloatingNorm(-1,2));
			domains.put("column2:x",new VariableDomainFloatingNorm(-3,4));domains.put("column2:y",new VariableDomainFloatingNorm(-3,0));
			domains.put("column3:x",new VariableDomainFloatingNorm(-2,5));domains.put("column3:y",new VariableDomainFloatingNorm(-2,1));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.3; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.2; tilingWidths[5] = 0.5;
			tilingWidths[6] = 0.3; tilingWidths[7] = 0.5;
			tilingWidths[8] = 0.3; tilingWidths[9] = 0.5;
			tilingWidths[10] = 0.3; tilingWidths[11] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_blocks_short){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(-11,0));domains.put("exit:y", new VariableDomain(-4,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-10,11));domains.put("block1:y",new VariableDomainFloatingNorm(-3,2));
			domains.put("block2:x",new VariableDomainFloatingNorm(-10,11));domains.put("block2:y",new VariableDomainFloatingNorm(-3,2));
			domains.put("block3:x",new VariableDomainFloatingNorm(-10,11));domains.put("block3:y",new VariableDomainFloatingNorm(-3,2));
			domains.put("block4:x",new VariableDomainFloatingNorm(-10,11));domains.put("block4:y",new VariableDomainFloatingNorm(-3,2));
			domains.put("column1:x",new VariableDomainFloatingNorm(-10,1));domains.put("column1:y",new VariableDomainFloatingNorm(-4,0));
			domains.put("column2:x",new VariableDomainFloatingNorm(-9,2));domains.put("column2:y",new VariableDomainFloatingNorm(-1,3));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.2; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.15; tilingWidths[5] = 0.3;
			tilingWidths[6] = 0.15; tilingWidths[7] = 0.3;
			tilingWidths[8] = 0.15; tilingWidths[9] = 0.3;
			tilingWidths[10] = 0.15; tilingWidths[11] = 0.3;
			tilingWidths[12] = 0.2; tilingWidths[13] = 0.5;
			tilingWidths[14] = 0.2; tilingWidths[15] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_blocks_short_2){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,11));domains.put("exit:y", new VariableDomain(-4,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-11,10));domains.put("block1:y",new VariableDomainFloatingNorm(-3,2));
			domains.put("block2:x",new VariableDomainFloatingNorm(-11,10));domains.put("block2:y",new VariableDomainFloatingNorm(-3,2));
			domains.put("block3:x",new VariableDomainFloatingNorm(-11,10));domains.put("block3:y",new VariableDomainFloatingNorm(-3,2));
			domains.put("block4:x",new VariableDomainFloatingNorm(-11,10));domains.put("block4:y",new VariableDomainFloatingNorm(-3,2));
			domains.put("column1:x",new VariableDomainFloatingNorm(-1,10));domains.put("column1:y",new VariableDomainFloatingNorm(-4,0));
			domains.put("column2:x",new VariableDomainFloatingNorm(-2,9));domains.put("column2:y",new VariableDomainFloatingNorm(-1,3));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.2; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.15; tilingWidths[5] = 0.3;
			tilingWidths[6] = 0.15; tilingWidths[7] = 0.3;
			tilingWidths[8] = 0.15; tilingWidths[9] = 0.3;
			tilingWidths[10] = 0.15; tilingWidths[11] = 0.3;
			tilingWidths[12] = 0.2; tilingWidths[13] = 0.5;
			tilingWidths[14] = 0.2; tilingWidths[15] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_blocks_short_3){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,11));domains.put("exit:y", new VariableDomain(-3,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-11,10));domains.put("block1:y",new VariableDomainFloatingNorm(-3,1));
			domains.put("block2:x",new VariableDomainFloatingNorm(-11,10));domains.put("block2:y",new VariableDomainFloatingNorm(-3,1));
			domains.put("block3:x",new VariableDomainFloatingNorm(-11,10));domains.put("block3:y",new VariableDomainFloatingNorm(-3,1));
			domains.put("block4:x",new VariableDomainFloatingNorm(-11,10));domains.put("block4:y",new VariableDomainFloatingNorm(-3,1));
			domains.put("column1:x",new VariableDomainFloatingNorm(-2,9));domains.put("column1:y",new VariableDomainFloatingNorm(-1,3));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.2; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.15; tilingWidths[5] = 0.3;
			tilingWidths[6] = 0.15; tilingWidths[7] = 0.3;
			tilingWidths[8] = 0.15; tilingWidths[9] = 0.3;
			tilingWidths[10] = 0.15; tilingWidths[11] = 0.3;
			tilingWidths[12] = 0.2; tilingWidths[13] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_blocks_pit){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomain(-17,0));domains.put("exit:y", new VariableDomain(-3,3));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-15,15));domains.put("block1:y",new VariableDomainFloatingNorm(-3,4));
			domains.put("block2:x",new VariableDomainFloatingNorm(-15,15));domains.put("block2:y",new VariableDomainFloatingNorm(-3,4));
			domains.put("block3:x",new VariableDomainFloatingNorm(-15,15));domains.put("block3:y",new VariableDomainFloatingNorm(-3,4));
			domains.put("block4:x",new VariableDomainFloatingNorm(-15,15));domains.put("block4:y",new VariableDomainFloatingNorm(-3,4));
			domains.put("column1:x",new VariableDomainFloatingNorm(-16,1));domains.put("column1:y",new VariableDomainFloatingNorm(-4,2));
			domains.put("column2:x",new VariableDomainFloatingNorm(-14,3));domains.put("column2:y",new VariableDomainFloatingNorm(-4,2));
			domains.put("column3:x",new VariableDomainFloatingNorm(-9,8));domains.put("column3:y",new VariableDomainFloatingNorm(-1,5));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.15; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.1; tilingWidths[5] = 0.3;
			tilingWidths[6] = 0.1; tilingWidths[7] = 0.3;
			tilingWidths[8] = 0.1; tilingWidths[9] = 0.3;
			tilingWidths[10] = 0.1; tilingWidths[11] = 0.3;
			tilingWidths[12] = 0.15; tilingWidths[13] = 0.3;
			tilingWidths[14] = 0.15; tilingWidths[15] = 0.3;
			tilingWidths[16] = 0.15; tilingWidths[17] = 0.3;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_step_min){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,1));domains.put("exit:y", new VariableDomainFloatingNorm(0,1));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("column1:x",new VariableDomainFloatingNorm(0,1));domains.put("column1:y",new VariableDomainFloatingNorm(0,-1));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.5; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.5; tilingWidths[5] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_step){//
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,7));domains.put("exit:y", new VariableDomainFloatingNorm(0,1));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("column1:x",new VariableDomainFloatingNorm(0,7));domains.put("column1:y",new VariableDomainFloatingNorm(0,-1));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.3; tilingWidths[1] = 0.1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.3; tilingWidths[5] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_reuse){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(-14,0));domains.put("exit:y", new VariableDomainFloatingNorm(0,5));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-13,14));domains.put("block1:y",new VariableDomainFloatingNorm(-5,4));
			domains.put("block2:x",new VariableDomainFloatingNorm(-13,14));domains.put("block2:y",new VariableDomainFloatingNorm(-5,4));
			domains.put("block3:x",new VariableDomainFloatingNorm(-13,14));domains.put("block3:y",new VariableDomainFloatingNorm(-5,4));
			domains.put("block4:x",new VariableDomainFloatingNorm(-13,14));domains.put("block4:y",new VariableDomainFloatingNorm(-5,4));
			domains.put("block5:x",new VariableDomainFloatingNorm(-13,14));domains.put("block5:y",new VariableDomainFloatingNorm(-5,4));
			domains.put("column1:x",new VariableDomainFloatingNorm(-14,0));domains.put("column1:y",new VariableDomainFloatingNorm(-1,4));
			domains.put("column2:x",new VariableDomainFloatingNorm(-11,3));domains.put("column2:y",new VariableDomainFloatingNorm(-4,1));
			domains.put("column3:x",new VariableDomainFloatingNorm(-7,7));domains.put("column3:y",new VariableDomainFloatingNorm(-4,1));
			domains.put("column4:x",new VariableDomainFloatingNorm(-5,9));domains.put("column4:y",new VariableDomainFloatingNorm(-2,3));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.15; tilingWidths[1] = 0.5;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.1; tilingWidths[5] = 0.25;
			tilingWidths[6] = 0.1; tilingWidths[7] = 0.25;
			tilingWidths[8] = 0.1; tilingWidths[9] = 0.25;
			tilingWidths[10] = 0.1; tilingWidths[11] = 0.25;
			tilingWidths[12] = 0.1; tilingWidths[13] = 0.25;
			tilingWidths[14] = 0.15; tilingWidths[15] = 0.3;
			tilingWidths[16] = 0.15; tilingWidths[17] = 0.3;
			tilingWidths[18] = 0.15; tilingWidths[19] = 0.3;
			tilingWidths[20] = 0.15; tilingWidths[21] = 0.3;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else if(s0 == initialState_useless_bc){//took care
			Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
			domains.put("exit:x", new VariableDomainFloatingNorm(0,10));domains.put("exit:y", new VariableDomainFloatingNorm(-1,0));
			domains.put("agent:dir",new VariableDomainFloatingNorm(0,1));domains.put("agent:holding",new VariableDomainFloatingNorm(0,1));
			domains.put("block1:x",new VariableDomainFloatingNorm(-10,10));domains.put("block1:y",new VariableDomainFloatingNorm(-1,1));
			domains.put("column1:x",new VariableDomainFloatingNorm(-8,2));domains.put("column1:y",new VariableDomainFloatingNorm(-1,0));
						
			TLNormalizedVarFeatBlockDude stateVariables = new TLNormalizedVarFeatBlockDude(domains);
			
			tileCoding = new TLTileCodingFeatures(stateVariables);
			
			double[] tilingWidths = new double[domains.size()];
			
			tilingWidths[0] = 0.3; tilingWidths[1] = 1;
			tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
			tilingWidths[4] = 0.15; tilingWidths[5] = 0.5;
			tilingWidths[6] = 0.3; tilingWidths[7] = 0.5;
			
			tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, numTilings, TilingArrangement.UNIFORM);
			
		}else{
			System.out.println("WARNING: there is no domain that can be associated to the selected map");
		}
		
		return tileCoding;
	}
	
	public static void delete(File toDelete){
		if(toDelete.exists()){
			String[] innerFiles = toDelete.list();
			for(String s : innerFiles){
				File currFile = new File(toDelete.getPath(),s);
				
				if(currFile.isDirectory())
					delete(currFile);
				else
					currFile.delete();
			}
			
			toDelete.delete();
		}	
	}
	
}
