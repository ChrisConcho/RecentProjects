using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WeaponScript : MonoBehaviour
{
    public float timeBtwShots;
    public GameObject projectile;


    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
    }

    public void fireWeapon()
    {
        Instantiate(projectile, transform.position, transform.rotation);
    }
}
