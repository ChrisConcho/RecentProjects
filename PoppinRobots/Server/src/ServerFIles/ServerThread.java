package ServerFIles;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.net.Socket;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class ServerThread extends Thread{

	Socket socket;
	// ObjectOutputStream os;
	// ObjectInputStream is;
	BufferedReader br;
	PrintWriter pw;
	Connection conn = null;
	Statement st = null;
	Statement tt = null;
	PreparedStatement ps = null;
	PreparedStatement tp = null;
	ResultSet rs = null;
	ResultSet ts = null;
	Example PlayersData;
	Player player;
	int player_id;
	public ServerThread(Socket s) {
		try {
			// to do --> store them somewhere, you will need them later 
			//is = new ObjectInputStream(s.getInputStream());
			br = new BufferedReader(new InputStreamReader(s.getInputStream()));
			//os = new ObjectOutputStream (s.getOutputStream());
			pw = new PrintWriter(s.getOutputStream());
			conn = DriverManager.getConnection("jdbc:mysql://localhost/PoppinRobots?user=root&password=root&serverTimezone=UTC");
			st = conn.createStatement();
			
			// rs = st.executeQuery("SELECT * from Student where fname='" + name + "'");
			
			socket = s;
			
			this.start();
			
			// to do --> complete the implementation for the constructor
			
			
			
			
			
			
		}catch(FileNotFoundException e) {
			System.out.println("No such file exists.");
		}catch (IOException ioe) {
			System.out.println("ioe in ServerThread constructor: " + ioe.getMessage());
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	
	}
	public void run() {
		Gson gson = new Gson();
		StringBuilder sb = new StringBuilder();
		String line;
		while (true) {
			
			try {
				
			//grab the jsonObjects from the finished game mode
			
				String command = br.readLine();
				// System.out.println(command);
				
				if (command.equals("Login")){ //read in that we are logging in from client.
					line = br.readLine();
					player = gson.fromJson(line, Player.class);
					String player_name = player.getNAME();
					//Select the skin and highscore for the client to use and send over as a jsonObject.
					ps = conn.prepareStatement("SELECT g.*, s.* ,IF(g.NAME =  '" + player_name + "' ,'true','false') "
							+ "AS FLAG  FROM Player g inner join Cosmetics s on g.PlayerID = s.ID ;");
					rs = ps.executeQuery();
					System.out.println("Passed SQL Query");
					boolean found_player = false;
					List<Integer> high_scores = new ArrayList<Integer>();
					List<Boolean> Cosmetics = new ArrayList<Boolean>();
					while(rs.next()) {
						boolean flag = rs.getBoolean("FLAG");
						int hs = rs.getInt("HighScore");
						if(flag) {
							player.setHighScore(hs);
							player.setMoney(rs.getInt("Money"));
							player.setPlayerID(rs.getInt("PlayerID"));
							player.setNewGamePlus(rs.getBoolean("NewGamePlus"));
							player.setWeaponID(rs.getInt("WeaponID"));
							Cosmetics.add(rs.getBoolean("GoldFit"));
							Cosmetics.add(rs.getBoolean("RedFit"));
							Cosmetics.add(rs.getBoolean("BlueFit"));
							Cosmetics.add(rs.getBoolean("BlackFit"));
							player.setCosmetic(Cosmetics);
							found_player = true;
						}
						high_scores.add(hs);
					}
					player.setAllHighScore(high_scores);
					String response;
					if ( found_player) {
						response = "Found";
						pw.print(response);
						pw.flush();
						System.out.println(response);
						response = gson.toJson(player,Player.class);
						pw.print(response);
						pw.flush();
						System.out.println(response);
					}
					else {
						response = "Absent";
						pw.print(response);
						pw.flush();
					}
					break;
				}
				if (command.equals("Register")) {
					line = br.readLine();
					player = gson.fromJson(line, Player.class);
					String NAME = player.getNAME();
					//ps = conn.prepareStatement("SELECT * FROM Player WHERE NAME = '123456'");
					ps = conn.prepareStatement("SELECT * FROM Player WHERE NAME = '" + NAME + "'");
					rs = ps.executeQuery();
					String response = "";
					if (rs.next()) {
						response = "Taken";
					}
					else {
						System.out.println("Registering user: " + NAME);
						ps = conn.prepareStatement("INSERT INTO Cosmetics ( GoldFit, RedFit,BlueFit,BlackFit) values (0,0,0,0);" );
						ps.executeUpdate();
						ps = conn.prepareStatement("INSERT INTO Player (NAME,HighScore, NewGamePlus,WeaponID,Money) values ('" + NAME + "', 0,false,0, 0);");
						ps.executeUpdate();
						response = "Valid";
					}
					pw.print(response);
					pw.flush();
					
					break;
				}
				if ( command.contentEquals("HighScores")) {
					List<String> users = new ArrayList<String>();
					List<Integer> high_scores = new ArrayList<Integer>();
					ps = conn.prepareStatement("SELECT NAME, HighScore FROM Player " ); //Select the skin and highscore for the client to use and send over as a jsonObject.
					rs = ps.executeQuery();
					while(rs.next()) {
						users.add(rs.getString("NAME"));
						high_scores.add(rs.getInt("HighScore"));
					}
					
					String user_json = gson.toJson(users);
					String hs_json = gson.toJson(high_scores);
					pw.print(user_json);
					pw.flush();
					pw.print(hs_json);
					pw.flush();
					break;
				}
				if (command.equals("Game Complete")) {
					line = br.readLine();
					player = gson.fromJson(line, Player.class);
					String Name = "NAME = '"+player.getNAME()+"',";
					String HighScore = "HighScore = "+player.getHighScore()+",";
					String NewGamePlus = "NewGamePlus = " + player.getNewGamePlus()+",";
					String WeaponID = "WeaponID = " + player.getWeaponID()+ ",";
					String Money = "Money = " + player.getMoney();
					
					ps=conn.prepareStatement("UPDATE Player SET "+Name+ HighScore + NewGamePlus + WeaponID + Money + "WHERE PlayerID = "+player_id+ ";");
					rs= ps.executeQuery();
					List<Boolean> Cosmos = player.getCosmetic();
					String White = "WhiteFit = '" + Cosmos.get(0) + "',";
					String Red = "RedFit = '" + Cosmos.get(1) + "',";
					String Blue = "BlueFit = '" + Cosmos.get(2) + "',";
					String Black = "BlackFit = '" + Cosmos.get(3) + "'";
					ps = conn.prepareStatement("UPDATE Cosmetics SET "+White + Red +Blue +Black+ "WHERE ID = "+ player_id +";");
					rs = ps.executeQuery();
					
					String response = "Succesfully Saved!";
					pw.print(response);
					pw.flush();
					break;
				}
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

}
