using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class hud : MonoBehaviour
{
    public Transform username;
    public Transform score;
    public Transform health;
    public Transform weapon;
    private GameObject Player;

    void Awake()
    {
        DontDestroyOnLoad(gameObject);
        Player = GameObject.FindGameObjectWithTag("Player");
        Debug.Log(Player);

        username.GetComponent<TextMeshProUGUI>().text = Player.GetComponent<PlayerScript>().playerData.username;
        score.GetComponent<TextMeshProUGUI>().text = "Score: " + Player.GetComponent<PlayerScript>().score;
        health.GetComponent<TextMeshProUGUI>().text = "Health: " + Player.GetComponent<PlayerScript>().health;
    }

    void Update()
    {
        username.GetComponent<TextMeshProUGUI>().text = Player.GetComponent<PlayerScript>().playerData.username;
        score.GetComponent<TextMeshProUGUI>().text = "Score: " + Player.GetComponent<PlayerScript>().score;
        health.GetComponent<TextMeshProUGUI>().text = "Health: " + Player.GetComponent<PlayerScript>().health;
    }

}
