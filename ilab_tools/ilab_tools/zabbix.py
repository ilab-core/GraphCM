import pyzabbix


def send_message(**kwargs):
    """
    Parameters
    ----------
    server : str, required
    host : str, required
    key : str, required
    message : str, required
    """
    try:
        request = pyzabbix.ZabbixSender(kwargs['server']).send(
            [pyzabbix.ZabbixMetric(kwargs['host'], kwargs['key'], kwargs['message'])])

        return request

    except Exception as e:
        return e
