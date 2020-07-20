FROM solr:latest

MAINTAINER Valentyna Fihurska "valentineburn@gmail.com"

COPY _default /opt/solr/server/solr/configsets/_default
COPY sample_products.xml /opt/solr/sample_products.xml

USER root

RUN chown -R solr:solr /opt/solr/server/solr/configsets/_default
RUN chown -R solr:solr /opt/solr/sample_products.xml

USER solr

#CMD ["/opt/solr/bin/solr", "-f"]
#CMD ["/opt/solr/bin/solr", "start"]
#CMD ["/opt/solr/bin/solr", "create", "-c", "sample_products"]
#CMD ["/opt/solr/bin/post", "-c", "sample_products", "sample_products.xml"]