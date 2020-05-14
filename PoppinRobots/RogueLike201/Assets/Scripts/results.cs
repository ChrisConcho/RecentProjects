using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class results : MonoBehaviour
{
    private Transform player;
    public Transform username;
    public Transform score;
    public Transform time;
    public Transform accuracy;
    public Transform final;
    public Transform message;

    // Update is called once per frame
    void Results(bool complete)
    {
        player = GameObject.FindGameObjectWithTag("Player").transform;

        // Server contact

        username.GetComponent<TextMeshProUGUI>().text = player.GetComponent<PlayerScript>().playerData.username;
        score.GetComponent<TextMeshProUGUI>().text = player.GetComponent<PlayerScript>().score.ToString();

        if (complete)
        {
            message.GetComponent<TextMeshProUGUI>().text = "Congratulations! Check if your score made it to the High Scores";
        }
        else
        {
            message.GetComponent<TextMeshProUGUI>().text = "Sorry! You lose";
        }

    }
}
