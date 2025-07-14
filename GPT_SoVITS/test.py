{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "67da8cbf",
   "metadata": {},
   "outputs": [
    {
     "ename": "",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31mRunning cells with 'sovits (Python 3.12.9)' requires the ipykernel package.\n",
      "\u001b[1;31m<a href='command:jupyter.createPythonEnvAndSelectController'>Create a Python Environment</a> with the required packages."
     ]
    }
   ],
   "source": [
    "from ruphon import RUPhon\n",
    "from ruaccent import RUAccent\n",
    "\n",
    "phonemizer = RUPhon()\n",
    "phonemizer = phonemizer.load(\"big\", workdir=\"./models\", device=\"GPU\")\n",
    "\n",
    "accentizer = RUAccent()\n",
    "accentizer.load(omograph_model_size='turbo3', use_dictionary=True, tiny_mode=False)\n",
    "\n",
    "input_text = \"я программирую на python.\"\n",
    "\n",
    "accented_text = accentizer.process_all(input_text)\n",
    "\n",
    "print(f\"Input: {input_text}\")\n",
    "print(f\"Accented: {accented_text}\")\n",
    "\n",
    "result = phonemizer.phonemize(accented_text, put_stress=True, stress_symbol=\"'\")\n",
    "\n",
    "print(f\"Phonemized: {result}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "sovits",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
