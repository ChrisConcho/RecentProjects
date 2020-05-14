using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using Newtonsoft.Json;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.SceneManagement;
using TMPro;

public class Client : MonoBehaviour
{
    // constructor for the client
    public Socket socket;
    public NetworkStream netStream;
    public GameObject playerObject;
    public TMP_InputField LI_UserInput;
    public TMP_InputField SU_UserInput;
    public Transform logInErrorText;
    public Transform signUpErrorText;


    public void Setup()
    {
        int port = 9999;

        socket = null;
        // create client object
        try
        {
            // Console.WriteLine("connecting to the socket");
            Debug.Log("connecting to the socket");

            // establish connection to the specified socket
            IPHostEntry ipHost = Dns.GetHostEntry(Dns.GetHostName());
            IPAddress ipAddress = ipHost.AddressList[0];
            IPEndPoint localEndPoint = new IPEndPoint(ipAddress, port);

            this.socket = new Socket(ipAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp);

            try
            {
                // attempt to connect to port
                // Console.WriteLine("Attempting to connect to port 9999");
                Debug.Log("Attempting to connect to port " + port);
                this.socket.Connect(localEndPoint);
                netStream = new NetworkStream(this.socket);
                // Console.WriteLine("Connected to port " + port);
                Debug.Log("Connected to port " + port);

            }
            catch (ArgumentNullException ane)
            {
                // Console.WriteLine("ArgumentNullException : {0}", ane.ToString());
                Debug.Log(ane.ToString());
            }
            catch (SocketException se)
            {
                // Console.WriteLine("SocketException : {0}", se.ToString());
                Debug.Log(se.ToString());
            }
            catch (Exception e)
            {
                // Console.WriteLine("Unexpected exception : {0}", e.ToString());
                Debug.Log(e.ToString());
            }
        }
        catch (Exception e)
        {
            // Console.WriteLine(e.ToString());
            Debug.Log(e.ToString());
        }
    }

    private void logIn(string username)
    {
        // create a new player and assign the text input as the username
        //Player player = gameObject.AddComponent<Player>();
        Player player = new Player();
        player.username = username;
        Console.WriteLine(username);

        // send the command 
        byte[] messageSent = Encoding.ASCII.GetBytes("Login\n");
        this.socket.Send(messageSent);

        // serialize the data to be sent across the socket
        string JSONresult = JsonConvert.SerializeObject(player) + "\n";
        messageSent = Encoding.ASCII.GetBytes(JSONresult);
        this.socket.Send(messageSent);

        // read in response from the server
        byte[] messageReceived = new byte[1024];
        int byteRecv = this.socket.Receive(messageReceived);

        // print response to the Console (for testing purposes)
        string response = Encoding.ASCII.GetString(messageReceived, 0, byteRecv);
        // Console.WriteLine("Message from Server -> {0}", response);
        Debug.Log(response);

        // found username
        if (response.Contains("Found"))
        {
            //logInErrorText.gameObject.SetActive(false);
            // read in JSON string and deserialize the object
            messageReceived = new byte[1024];
            byteRecv = this.socket.Receive(messageReceived);
            response = Encoding.ASCII.GetString(messageReceived, 0, byteRecv);

            // assign the object to the player object
            player = JsonConvert.DeserializeObject<Player>(response);
            Debug.Log("Found the profile");

            //Creates playerObject with playerData passed in.
            Instantiate(playerObject, Vector3.up, Quaternion.identity);
            GameObject Player = GameObject.FindGameObjectWithTag("Player");
            Player.GetComponent<PlayerScript>().playerData = player;

            SceneManager.LoadScene("room_start");

        }
        // did not find username
        else if (response.Contains("Absent"))
        {
            //logInErrorText.gameObject.SetActive(true);
            // MENU SHOULD PROMPT THE USER TO ENTER ANOTHER USERNAME
            // OR SUGGEST REGISTERING AS A NEW USER
            // OR HAVE THE BACK OPTION
        }
    }

    private void registerUser(string username)
    {
        // Console.WriteLine("registering user " + username);
        Debug.Log("registering user " + username);
        //Player player = gameObject.AddComponent<Player>();
        Player player = new Player();
        player.username = username;

        // send command to the server
        byte[] messageSent = Encoding.ASCII.GetBytes("Register\n");
        this.socket.Send(messageSent);

        // serialize the data to be sent across the socket
        string JSONresult = JsonConvert.SerializeObject(player) + "\n";
        messageSent = Encoding.ASCII.GetBytes(JSONresult);
        Debug.Log(JSONresult);
        this.socket.Send(messageSent);

        byte[] messageReceived = new byte[1024];
        int byteRecv = this.socket.Receive(messageReceived);

        // print response to the Console (for testing purposes)
        string response = Encoding.ASCII.GetString(messageReceived, 0, byteRecv);

        Debug.Log("Received: " + response);

        if (response.Contains("Valid"))
        {
            //signUpErrorText.gameObject.SetActive(false);
            //messageReceived = new byte[1024];
            //int byteRecv2 = this.socket.Receive(messageReceived);
            //response = Encoding.ASCII.GetString(messageReceived, 0, byteRecv2);

            // assign to the new player
            //this.player = JsonConvert.DeserializeObject<Player>(response);

            //Creates playerObject with playerData passed in.
            Instantiate(playerObject, Vector3.up, Quaternion.identity);
            GameObject Player = GameObject.FindGameObjectWithTag("Player");
            Player.GetComponent<PlayerScript>().playerData = player;

            Debug.Log("Valid Registration");

            SceneManager.LoadScene("room_start");

            // send this.player to game
        }
        else if (response.Contains("Taken"))
        {
            Debug.Log("Username Already Exists");
            //signUpErrorText.gameObject.SetActive(true);
            // MENU SHOULD PROMPT THE USER TO ENTER ANOTHER USERNAME
            // OR PROCEED TO LOGIN MENU
        }
    }

    public ArrayList[] getHighScores()
    {
        ArrayList userList = new ArrayList();
        ArrayList scoreList = new ArrayList();

        byte[] messageSent = Encoding.ASCII.GetBytes("HighScores\n");
        this.socket.Send(messageSent);

        byte[] messageReceived = new byte[1024];
        int byteRecv = this.socket.Receive(messageReceived);
        string response = Encoding.ASCII.GetString(messageReceived, 0, byteRecv);

        userList = JsonConvert.DeserializeObject<ArrayList>(response);

        messageReceived = new byte[1024];
        byteRecv = this.socket.Receive(messageReceived);
        response = Encoding.ASCII.GetString(messageReceived, 0, byteRecv);

        scoreList = JsonConvert.DeserializeObject<ArrayList>(response);

        ArrayList[] retList = { userList, scoreList };
        /*
        for (int i = 0; i < userList.Count; i++)
        {
            KeyValuePair<string, int> pair = new KeyValuePair<string, int>((string)userList[i], (int)scoreList[i]);
            retList.Add(pair);
            Debug.Log(pair.First + " " + pair.Second);
        }
        */
        return retList;
    }

    /*
    public void updateJSON(int prevScore, bool gameFinished, int weaponID, int money)
    {
        // Console.WriteLine("updating user information for user: " + player.username);
        Debug.Log("updating user information for user: " + player.username);

        // send command to the server
        byte[] messageSent = Encoding.ASCII.GetBytes("Login");
        this.socket.Send(messageSent);

        // update high-score if it is the new highest
        if (prevScore > this.player.AllhighScore[0])
        {
            this.player.highScore = prevScore;
        }
        // update list of total scores
        this.player.AllhighScore.Add(prevScore);
        // sort the list now

        // update new game plus if necessary
        if (gameFinished)
        {
            this.player.newGamePlus = true;

            // update weaponid
            this.player.weaponID = weaponID;

            // update money (does money get reset if you do not win the game)
            this.player.money = money;
        }

        // update cosmetics if applicable
        if (gameFinished)
        {
            System.Collections.IList list = player.cosmetic;
            for (int i1 = 0; i1 < list.Count; i1++)
            {
                int i = (int)list[i1];
                this.player.cosmetic[i] = true;
            }
        }

        // serialize the object and send to the server
        string JSONresult = JsonConvert.SerializeObject(this.player);
        messageSent = Encoding.ASCII.GetBytes(JSONresult);
        this.socket.Send(messageSent);

    }
    */

    public void LogInFunction()
    {
        string username = LI_UserInput.text;

        if (username != null)
        {

            Client client = GetComponentInParent<Client>();
            client.Setup();
            client.logIn(username);
        }
    }

    public void SignUpFunction()
    {
        string username = SU_UserInput.text;

        if (username != null)
        {
            Client client = GetComponentInParent<Client>();
            client.Setup();
            client.registerUser(username);
        }
    }
}

