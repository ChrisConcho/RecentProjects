using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DefaultOff : MonoBehaviour
{
    public Transform t;

    // Start is called before the first frame update
    void Start()
    {
        t.gameObject.SetActive(false);
    }
}
