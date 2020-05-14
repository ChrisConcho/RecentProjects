using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BackgroundScript : MonoBehaviour
{
    private Vector3 bgPosition;
    private Vector3 bgVelocity;
    private double x_max;
    private double x_min;
    private double y_max;
    private double y_min;

    // Start is called before the first frame update
    void Start()
    {
        bgPosition = new Vector3(0.0f, 0.0f, 0.0f);
        bgVelocity = new Vector3(0.01f, 0.01f, 0.01f);
        x_max = 1.6f;
        x_min = -1.6f;
        y_max = 3.5f;
        y_min = -3.5f;
    }

    // Update is called once per frame
    void Update()
    {
        if (bgPosition.x < x_min)
        {
            bgVelocity.x = 0.01f;
        }
        else if (bgPosition.x > x_max)
        {
            bgVelocity.x = -0.01f;
        }

        if (bgPosition.y < y_min)
        {
            bgVelocity.y = 0.01f;
        }
        else if (bgPosition.y > y_max)
        {
            bgVelocity.y = -0.01f;
        }

        bgPosition.x += bgVelocity.x;
        bgPosition.y += bgVelocity.y;

        transform.position = bgPosition;
    }
}