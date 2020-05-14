using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WeaponSwitchScript : MonoBehaviour
{
    private float fireRate;
    public int selectedWeapon = 0;
    public Transform[] prefabs;
    // Start is called before the first frame update

    void Start()
    {
        SelectWeapon();
    }

    // Update is called once per frame
    void Update()
    {

        int previousWeapon = selectedWeapon;
        /**
        if (Input.GetAxis("Mouse ScrollWheel") > 0f)
        {
            if (selectedWeapon >= transform.childCount-1)
            {
                selectedWeapon = 0;
            }
            else
            {
                selectedWeapon++;
            }
        }
        if(Input.GetAxis("Mouse ScrollWheel") < 0f)
        {
            if (selectedWeapon <= 0)
            {
                selectedWeapon = transform.childCount-1;
            }
            else
            {
                selectedWeapon--;
            }
        }
        **/
        if (fireRate <= 0.0f)
        {
            //if left click is pressed/held
            if (Input.GetMouseButton(0))
            {
                fireWeapon();
            }
        }
        else
        {
            fireRate -= Time.deltaTime;
        }


        if (selectedWeapon != previousWeapon)
        {
            SelectWeapon();
        }
    }

    void SelectWeapon()
    {
        int i = 0;
        foreach(Transform weapon in transform)
        {
            if(i == selectedWeapon)
            {
                weapon.gameObject.SetActive(true);
            }
            else
            {
                weapon.gameObject.SetActive(false);
            }
            i++;
        }
        fireRate = transform.GetChild(selectedWeapon).GetComponent<WeaponScript>().timeBtwShots;
    }

    public void setWeapon(int weaponId)
    {
        int previousWeapon = selectedWeapon;
        if(weaponId >= 0)
        {
            selectedWeapon = weaponId;
        }

        Instantiate(prefabs[previousWeapon],transform.position, Quaternion.LookRotation(Vector3.forward,Vector3.up));
    }

    public void fireWeapon()
    {
        transform.GetChild(selectedWeapon).GetComponent<WeaponScript>().fireWeapon();
        fireRate = transform.GetChild(selectedWeapon).GetComponent<WeaponScript>().timeBtwShots;
    }
}
