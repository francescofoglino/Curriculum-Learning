//***********************************************************HELLO BLOCK DUDE!***************************************************************************

package myProj;

import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
import burlap.domain.singleagent.blockdude.state.BlockDudeAgent;
import burlap.domain.singleagent.blockdude.state.BlockDudeCell;
import burlap.domain.singleagent.blockdude.state.BlockDudeMap;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.visualizer.Visualizer;
import burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor; //temp


public class HelloBlockDude {

	public static void main(String[] args) {

		BlockDude bd = new BlockDude(); //Initialises a world with a maximum 25x25 dimensionality and actions that use semi-deep state copies.
		
		SADomain domain = bd.generateDomain(); //generate the grid world domain

		//setup initial state
		//State s = new BlockDudeState(new BlockDudeAgent(), 
		//							 new BlockDudeMap(10, 10), 
		//							 new BlockDudeCell(9, 1, BlockDude.CLASS_EXIT, "exit"), 
		//							 new BlockDudeCell(7, 1, BlockDude.CLASS_BLOCK, "obstacle"));

		State s = BlockDudeLevelConstructor.getLevel2(domain);
		
		//create visualizer and explorer
		Visualizer v = BlockDudeVisualizer.getVisualizer(20, 20);// max dimension of the map
		VisualExplorer exp = new VisualExplorer(domain, v, s);

		//set control keys to use w-s-a-d
		exp.addKeyAction("w", BlockDude.ACTION_UP, "");
		exp.addKeyAction("a", BlockDude.ACTION_WEST, "");
		exp.addKeyAction("d", BlockDude.ACTION_EAST, "");
		exp.addKeyAction("s", BlockDude.ACTION_PICKUP, "");
		exp.addKeyAction("x", BlockDude.ACTION_PUT_DOWN, "");

		exp.initGUI();

	}

}