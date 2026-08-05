


from __future__ import annotations
import sys
from pathlib import Path
import os


# In[8]:


try:
    _ROOT = Path(__file__).resolve().parent
except NameError:
    _ROOT = Path(os.getcwd()).resolve()

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# In[9]:


import Coupling
import UWM


# In[10]:


__all__ = [
    "QuasiStaticFSI",
    "QuasiStaticHistory",
    "updated_weight_form_find",
    "UWMResult",
]


# In[ ]:




