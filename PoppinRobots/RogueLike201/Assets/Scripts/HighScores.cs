using System.Collections;
using System;
using UnityEngine;
using TMPro;

public class HighScores : MonoBehaviour
{
    public Transform entryContainer;
    public Transform entryTemplate;

    public void populate()
    {
        entryTemplate.gameObject.SetActive(false);

        float templateHeight = 25f;

        Client client = GetComponentInParent<Client>();
        client.Setup();

        ArrayList[] highScores = client.getHighScores();
        // stringArray.Select(Int32.Parse).ToList();
        // get some vector/array with 5 values

        int size = highScores[0].Count;
        int max_idx, val_1, val_2;
        string temp_str;

        for (int i = 0; i < size-1; i++)
        {
            max_idx = i;

            for (int j = i + 1; j < size; j++)
            {
                val_1 = Int32.Parse(highScores[1][j].ToString());
                val_2 = Int32.Parse(highScores[1][max_idx].ToString());

                if (val_1 > val_2)
                    max_idx = j;
            }
            val_1 = Int32.Parse(highScores[1][i].ToString());

            int temp = val_1;
            highScores[1][i] = highScores[1][max_idx];
            highScores[1][max_idx] = temp;

            temp_str = highScores[0][i].ToString();
            highScores[0][i] = highScores[0][max_idx];
            highScores[0][max_idx] = temp_str;
        }

        for (int i = 0; (i < size && i < 5); i++)
        {
            Transform entryTransform = Instantiate(entryTemplate, entryContainer);
            RectTransform entryRectTransform = entryTransform.GetComponent<RectTransform>();
            entryRectTransform.anchoredPosition = new Vector2(0, -10 - templateHeight * i);
            entryTransform.gameObject.SetActive(true);
            entryTransform.Find("UserEntry").GetComponent<TextMeshProUGUI>().text = highScores[0][i].ToString();
            entryTransform.Find("ScoreEntry").GetComponent<TextMeshProUGUI>().text = highScores[1][i].ToString();
            Debug.Log(highScores[0][i] + " " + highScores[1][i]);
        }
    }

    IEnumerator WaitCoroutine(float x)
    {
        yield return new WaitForSeconds(x);
    }
}
