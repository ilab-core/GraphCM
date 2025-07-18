import slack_sdk
import slack_sdk.errors


def send_message(**kwargs):
    """
    Parameters
    ----------
    qauth_token : str, required
    channel_id : str, required
    message : str, required
    """
    web_client = slack_sdk.WebClient(token=kwargs['qauth_token'])

    try:
        web_client.chat_postMessage(channel=kwargs['channel_id'], text=kwargs['message'])
    except slack_sdk.errors.SlackApiError as e:
        assert e.response['error']
