using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerManagmentScript : MonoBehaviour
{
    // Start is called before the first frame update


    void Start()
    {
        //Debug.Log(transform.position);
        GameObject player = GameObject.FindGameObjectWithTag("Player");
        player.transform.position = transform.position;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
