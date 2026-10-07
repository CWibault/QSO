from qso.agents.aso import ASOAgent
from qso.agents.hiql import HIQLAgent
from qso.agents.iql import IQLAgent
from qso.agents.opal import OPALAgent
from qso.agents.qc import QCAgent
from qso.agents.qso import QSOAgent

agents = dict(
    aso=ASOAgent,
    hiql=HIQLAgent,
    iql=IQLAgent,
    opal=OPALAgent,
    qc=QCAgent,
    qso=QSOAgent,
)
