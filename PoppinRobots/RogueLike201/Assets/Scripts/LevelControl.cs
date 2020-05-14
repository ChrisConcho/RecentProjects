using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class LevelControl : MonoBehaviour
{
    public Animator transition;
    public float transitionTime;
    public string levelName;
    public GameObject playerObject;

    private void OnTriggerEnter2D(Collider2D collision)
    {
        if (collision.CompareTag("Player"))
        {
            StartCoroutine(loadLevel());
        }
    }

    public void OnMouseClickGuest() {
        //Player player = gameObject.AddComponent<Player>();
        Player player = new Player();
        player.username = "Guest";

        Instantiate(playerObject, Vector3.up, Quaternion.identity);
        GameObject Player = GameObject.FindGameObjectWithTag("Player");
        Player.GetComponent<PlayerScript>().playerData = player;

        SceneManager.LoadScene("room_start"); 
    }

    IEnumerator loadLevel()
    {
        transition.SetTrigger("Start");

        yield return new WaitForSeconds(transitionTime);

        SceneManager.LoadScene(levelName);

    }
}
