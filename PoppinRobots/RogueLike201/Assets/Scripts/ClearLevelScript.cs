using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ClearLevelScript : MonoBehaviour
{
    // Start is called before the first frame update
    private SpriteRenderer sr;
    private CircleCollider2D cc2d;
    void Start()
    {
        sr = GetComponent<SpriteRenderer>();
        cc2d = GetComponent<CircleCollider2D>();
        sr.enabled = false;
        cc2d.enabled = false;
    }

    // Update is called once per frame
    void Update()
    {
        //Debug.Log(GameObject.FindGameObjectsWithTag("Enemy").Length);
        if (GameObject.FindGameObjectsWithTag("Enemy").Length == 0)
        {
            sr.enabled = true;
            cc2d.enabled = true;

        }
    }
}
