using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class Score : MonoBehaviour
{
    public Transform hud;
	public Transform score;
	//public Transform player;
	public System.DateTime startTime;
	private int scoreCount;

	void Start(){
		startTime = System.DateTime.UtcNow;
    }
    public void UpdateKill(string enemy)
    {
    	if ( enemy == "SmallFlying"){
    		scoreCount += 25;
    	}
    	if( enemy == "Humanoid"){
    		scoreCount +=50;
    	}

    	if( enemy == "Sniper"){
    		scoreCount += 100;
    	}

    	if( enemy == "Bomber"){

    		scoreCount += 150;
    	}

    	if ( enemy == "Tank"){

    		scoreCount += 125;
    	}

    	if ( enemy == "Boss"){

    		scoreCount += 500;
    	}
    	score.GetComponent<TextMeshProUGUI>().text = "Score: " + scoreCount;

    }
    public void GameOver(){
    	System.TimeSpan ts = System.DateTime.UtcNow - startTime;
    	scoreCount += 1000000/ts.Seconds;
    	score.GetComponent<TextMeshProUGUI>().text = "Score: " + scoreCount;
    }

    
    
}
