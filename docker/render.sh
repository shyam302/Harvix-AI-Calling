#!/bin/sh
# Fill ${VARS} in the config templates from docker/.env, then start Asterisk.
# Only the variables listed below are replaced, so Asterisk's own ${EXTEN} etc. stay intact.
set -e
VARS='${SIP_PORT} ${RTP_START} ${RTP_END} ${PHONE_1001_PASSWORD} ${PHONE_1002_PASSWORD}
      ${ARI_BIND} ${ARI_PORT} ${ARI_USERNAME} ${ARI_PASSWORD} ${BOT_EXTENSION} ${STASIS_APP}'
for f in /templates/*.conf; do
    envsubst "$VARS" < "$f" > "/etc/asterisk/$(basename "$f")"
done
if [ -n "$EXTERNAL_IP" ]; then
    sed -i "/^\[transport-udp\]/a external_media_address=$EXTERNAL_IP\nexternal_signaling_address=$EXTERNAL_IP" /etc/asterisk/pjsip.conf
fi
# The logs bind mount hides the image's cdr-csv folder; recreate it.
mkdir -p /var/log/asterisk/cdr-csv /var/log/asterisk/cdr-custom
chown -R "${PUID:-1000}:${PGID:-1000}" /var/log/asterisk
exec /usr/local/bin/entrypoint.sh "$@"
