#!/bin/sh
# Fill ${VARS} in the config templates from docker/.env, then start Asterisk.
# Only the variables listed below are replaced, so Asterisk's own ${EXTEN} etc. stay intact.
set -e
VARS='${CODECS} ${SIP_PORT} ${RTP_START} ${RTP_END} ${ARI_BIND} ${ARI_PORT} ${ARI_USERNAME} ${ARI_PASSWORD} ${BOT_EXTENSION} ${STASIS_APP}'
for f in /templates/*.conf; do
    envsubst "$VARS" < "$f" > "/etc/asterisk/$(basename "$f")"
done
# Phone accounts: PHONES="1001:pass1,1002:pass2,…" (extension:password).
# Empty PHONES = the old PHONE_1001_PASSWORD / PHONE_1002_PASSWORD settings.
PHONES="${PHONES:-1001:${PHONE_1001_PASSWORD:-1001},1002:${PHONE_1002_PASSWORD:-1002}}"
count=0
for entry in $(echo "$PHONES" | tr ',' ' '); do
    ext="${entry%%:*}"
    pass="${entry#*:}"
    case "$ext" in
        ''|*[!0-9]*) echo "render.sh: skipping bad PHONES entry '$entry' (extension must be digits)" >&2; continue ;;
    esac
    if [ -z "$pass" ] || [ "$pass" = "$entry" ]; then
        echo "render.sh: skipping '$ext': no password (use extension:password)" >&2; continue
    fi
    cat >> /etc/asterisk/pjsip.conf <<CONF

[$ext]
type=auth
auth_type=userpass
username=$ext
password=$pass

[$ext](aor-tpl)

[$ext](phone)
auth=$ext
aors=$ext
CONF
    count=$((count + 1))
done
echo "render.sh: $count phone account(s) from PHONES"

if [ -n "$EXTERNAL_IP" ]; then
    sed -i "/^\[transport-udp\]/a external_media_address=$EXTERNAL_IP\nexternal_signaling_address=$EXTERNAL_IP" /etc/asterisk/pjsip.conf
fi
# Keep astdb (phones added from the web page) on the asterisk-db volume.
sed -i 's|^\[directories\](!)|[directories]|; s|^astdbdir => .*|astdbdir => /var/lib/asterisk/db|' /etc/asterisk/asterisk.conf
mkdir -p /var/lib/asterisk/db
chown -R "${PUID:-1000}:${PGID:-1000}" /var/lib/asterisk/db

# The logs bind mount hides the image's cdr-csv folder; recreate it.
mkdir -p /var/log/asterisk/cdr-csv /var/log/asterisk/cdr-custom /var/spool/asterisk/recording
chown -R "${PUID:-1000}:${PGID:-1000}" /var/spool/asterisk
chown -R "${PUID:-1000}:${PGID:-1000}" /var/log/asterisk
exec /usr/local/bin/entrypoint.sh "$@"
