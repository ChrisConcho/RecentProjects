package ServerFIles;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;


public class Server {
	public ServerSocket serverSocket;

	public Server(int port) {
		// to do --> implement your constructor
			
			try {
				serverSocket = new ServerSocket(port);
			
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Server server= new Server(9999);
		Connection conn = null;
		Connection conn2 = null;
		Statement database_statement = null;
		Statement Cosmos_statement =null;
		Statement Player_statement = null;
		Statement use_statement = null;
		
		System.out.println("Server is running");
		try {
		
			
			String send_out = "CREATE DATABASE IF NOT EXISTS PoppinRobots;";
			String use_robots=" USE PoppinRobots;";
			String cosmetic_table = "CREATE TABLE IF NOT EXISTS Cosmetics ( ID INT(2) PRIMARY KEY NOT NULL AUTO_INCREMENT, GoldFit BOOLEAN NOT NULL, RedFit BOOLEAN NOT NULL, BlueFit BOOLEAN NOT NULL, BlackFit BOOLEAN NOT NULL );";
			String player_table = "CREATE TABLE IF NOT EXISTS Player ( PlayerID INT(6) PRIMARY KEY NOT NULL AUTO_INCREMENT, NAME VARCHAR(20) NOT NULL, HighScore INT(20) NOT NULL, NewGamePlus BOOLEAN NOT NULL, WeaponID INT(1) NOT NULL, Money INT(10) NOT NULL, FOREIGN KEY fk5(PlayerID) REFERENCES Cosmetics(ID)); ";
	
			conn = DriverManager.getConnection("jdbc:mysql://localhost/?user=root&password=root&serverTimezone=UTC");
			
			database_statement = conn.createStatement();
			database_statement.executeUpdate(send_out);
			
			conn2 = DriverManager.getConnection("jdbc:mysql://localhost/PoppinRobots?user=root&password=root&serverTimezone=UTC");
			use_statement = conn2.createStatement();
			Cosmos_statement = conn2.createStatement();
			Player_statement = conn2.createStatement();
			
			use_statement.executeUpdate(use_robots);
			Cosmos_statement.executeUpdate(cosmetic_table);
			Player_statement.executeUpdate(player_table);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		
		while(true) {
			Socket socket;
			try {
				socket = server.serverSocket.accept();
			
				ServerThread serverThread = new ServerThread(socket);
				System.out.println("Given a Thread");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		
		}
	}

}
